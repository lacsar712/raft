// Copyright IBM Corp. 2026
// SPDX-License-Identifier: MPL-2.0

package raft

import (
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/hashicorp/go-metrics"
)

// Kind of work represented by a scheduled entry.
const (
	scheduledCommand = iota
	scheduledBarrier
	scheduledConfiguration
)

// scheduledApply tracks a single committed entry handed to the FSM.
type scheduledApply struct {
	req *commitTuple

	index uint64
	term  uint64
	kind  int

	// dep is the index through which all earlier entries must have
	// settled before this entry may run. It includes waits forced by
	// preceding barriers and configuration entries.
	dep uint64

	// failDep is the dependency declared by the FSM (commands only).
	// Failure propagation follows only the declared dependency edges, so
	// an independent entry that merely had to wait behind a barrier or a
	// configuration entry is not aborted by an unrelated failure.
	failDep uint64

	// counted reports whether settling this entry advances the snapshot
	// index/term. Configuration entries that the FSM does not implement
	// ConfigurationStore for are not counted, matching the behavior of the
	// serial apply path.
	counted bool

	done bool
}

// fsmApplyScheduler runs on the FSM goroutine and admits committed entries
// in index order. Each admitted entry is executed by its own goroutine,
// which parks until the entry's dependency prefix has settled, allowing
// independent entries to be applied concurrently while ordering dependent
// ones. All scheduler state is touched either by the admitting FSM
// goroutine or by a worker goroutine while holding mu.
type fsmApplyScheduler struct {
	r           *Raft
	fsm         DependencyFSM
	configStore ConfigurationStore

	mu   sync.Mutex
	cond *sync.Cond

	// states holds entries admitted but not yet part of the contiguous
	// settled prefix. Indexes absent from the map are indexes that never
	// reach the FSM (no-ops, deprecated peer entries); as long as they are
	// not past the admitted watermark they count as settled gaps.
	states map[uint64]*scheduledApply

	// admitted is the highest index handed to the scheduler so far.
	// settled is the highest index through which every admitted entry (or
	// gap) has finished.
	admitted uint64
	settled  uint64

	// failed is the ascending list of indexes whose FSM.Apply returned an
	// error. Entries stay here until a Restore resets the scheduler; they
	// are used to abort later entries that depend on a failed prefix.
	failed []uint64

	// serialIndex is the index of the most recent barrier/configuration
	// entry; every later command is forced to wait at least until that
	// entry has settled.
	serialIndex uint64

	// snapIndex/snapTerm describe the highest counted, settled entry and
	// are used as snapshot metadata.
	snapIndex uint64
	snapTerm  uint64
}

func newFSMApplyScheduler(r *Raft) *fsmApplyScheduler {
	s := &fsmApplyScheduler{
		r:           r,
		fsm:         r.fsm.(DependencyFSM),
		configStore: nil,
		states:      make(map[uint64]*scheduledApply),
	}
	if cs, ok := r.fsm.(ConfigurationStore); ok {
		s.configStore = cs
	}
	s.cond = sync.NewCond(&s.mu)
	return s
}

// admit takes ownership of a committed entry (in commit order) and starts
// its worker goroutine. It runs on the FSM goroutine.
func (s *fsmApplyScheduler) admit(req *commitTuple) {
	idx := req.log.Index

	st := &scheduledApply{
		req:   req,
		index: idx,
		term:  req.log.Term,
	}

	switch req.log.Type {
	case LogBarrier:
		// Barriers drain every earlier entry and serialize the stream.
		st.kind = scheduledBarrier
		st.dep = idx - 1
		st.counted = true
		s.serialIndex = idx

	case LogConfiguration:
		// Membership changes always run with no other Apply in flight and
		// every later entry waits for them.
		st.kind = scheduledConfiguration
		st.dep = idx - 1
		st.counted = s.configStore != nil
		s.serialIndex = idx

	default:
		st.kind = scheduledCommand
		st.counted = true

		dep := s.fsm.ApplyDependency(req.log)
		if dep >= idx {
			s.r.logger.Warn("clamping ApplyDependency result to the preceding index",
				"index", idx, "dependency", dep)
			dep = idx - 1
		}
		st.failDep = dep
		if dep < s.serialIndex {
			dep = s.serialIndex
		}
		st.dep = dep
	}

	s.mu.Lock()
	s.states[idx] = st
	s.admitted = idx
	// Roll the settled frontier over any indexes ahead of it that will
	// never reach the FSM (gaps); without this an entry waiting for a
	// gap could block forever.
	s.advanceLocked()
	s.cond.Broadcast()
	s.mu.Unlock()

	s.r.goFunc(func() { s.runEntry(st) })
}

// runEntry executes a single admitted entry after waiting for its
// dependency prefix to settle.
func (s *fsmApplyScheduler) runEntry(st *scheduledApply) {
	s.mu.Lock()
	for {
		// A command whose dependency prefix contains a failed entry is
		// aborted instead of being applied. Barriers and configuration
		// entries are Raft-infrastructure operations and always run once
		// their prefix settles, and independent commands still proceed.
		if st.kind == scheduledCommand && s.hasFailedAtOrBeforeLocked(st.failDep) {
			s.mu.Unlock()
			s.finish(st, false, nil, true)
			return
		}
		if s.settled >= st.dep {
			break
		}
		s.cond.Wait()
	}
	s.mu.Unlock()

	var resp interface{}
	switch st.kind {
	case scheduledCommand:
		start := time.Now()
		resp = s.fsm.Apply(st.req.log)
		metrics.MeasureSince([]string{"raft", "fsm", "apply"}, start)

	case scheduledConfiguration:
		if s.configStore != nil {
			start := time.Now()
			s.configStore.StoreConfiguration(st.req.log.Index, DecodeConfiguration(st.req.log.Data))
			metrics.MeasureSince([]string{"raft", "fsm", "store_config"}, start)
		}
	}

	s.finish(st, true, resp, false)
}

// finish records the outcome of an entry, settles the contiguous prefix as
// far as possible and answers the entry's future.
func (s *fsmApplyScheduler) finish(st *scheduledApply, ran bool, resp interface{}, aborted bool) {
	future := st.req.future

	s.mu.Lock()
	if ran && st.kind == scheduledCommand {
		if err, ok := resp.(error); ok && err != nil {
			s.failed = append(s.failed, st.index)
			sort.Slice(s.failed, func(i, j int) bool { return s.failed[i] < s.failed[j] })
		}
	}
	st.done = true
	s.advanceLocked()
	s.cond.Broadcast()
	s.mu.Unlock()

	if future != nil {
		switch {
		case aborted:
			s.r.logger.Warn("not applying entry because an entry it depends on failed",
				"index", st.index, "dependency", st.dep)
			future.respond(fmt.Errorf("%w: index %d depends on a failed entry at or before index %d",
				ErrDependencyAborted, st.index, st.failDep))

		case st.kind == scheduledCommand:
			future.response = resp
			future.respond(nil)

		default:
			future.respond(nil)
		}
	}
}

// hasFailedAtOrBeforeLocked reports whether any failed Apply sits in the
// prefix up to and including dep.
func (s *fsmApplyScheduler) hasFailedAtOrBeforeLocked(dep uint64) bool {
	for _, idx := range s.failed {
		if idx <= dep {
			return true
		}
		// s.failed is sorted in ascending order.
		break
	}
	return false
}

// advanceLocked moves the settled frontier across done entries and over
// gaps as far as possible, updating snapshot metadata.
func (s *fsmApplyScheduler) advanceLocked() {
	for {
		next := s.settled + 1
		if next > s.admitted {
			return
		}
		st, ok := s.states[next]
		if !ok {
			// An index that will never be handed to the FSM (e.g. a no-op
			// log); next <= admitted proves it is not merely unadmitted.
			s.settled = next
			continue
		}
		if !st.done {
			return
		}
		if st.counted {
			s.snapIndex = next
			s.snapTerm = st.term
		}
		delete(s.states, next)
		s.settled = next
	}
}

// waitIdle blocks until every admitted entry has settled, so no FSM.Apply
// is in flight. Runs on the FSM goroutine; no new entry can be admitted
// while it is blocked.
func (s *fsmApplyScheduler) waitIdle() {
	s.mu.Lock()
	defer s.mu.Unlock()
	for s.settled < s.admitted {
		s.cond.Wait()
	}
}

// snapshot drains all in-flight applies before asking the FSM for a
// snapshot, guaranteeing a consistent view.
func (s *fsmApplyScheduler) snapshot(req *reqSnapshotFuture) {
	s.waitIdle()

	if s.snapIndex == 0 {
		req.respond(ErrNothingNewToSnapshot)
		return
	}

	start := time.Now()
	snap, err := s.r.fsm.Snapshot()
	metrics.MeasureSince([]string{"raft", "fsm", "snapshot"}, start)

	req.index = s.snapIndex
	req.term = s.snapTerm
	req.snapshot = snap
	req.respond(err)
}

// restore drains all in-flight applies before restoring the FSM and then
// rebases the scheduler on the restored index.
func (s *fsmApplyScheduler) restore(req *restoreFuture) {
	s.waitIdle()

	index, term, err := s.r.restoreFSM(req)
	if err != nil {
		req.respond(err)
		return
	}

	s.mu.Lock()
	s.states = make(map[uint64]*scheduledApply)
	s.failed = nil
	s.serialIndex = 0
	s.admitted = index
	s.settled = index
	s.snapIndex = index
	s.snapTerm = term
	s.cond.Broadcast()
	s.mu.Unlock()

	req.respond(nil)
}

// runFSMParallel is the FSM goroutine loop used when the FSM implements
// DependencyFSM. It keeps the same channels and message ordering as the
// serial runFSM; only the application of committed entries changes.
func (r *Raft) runFSMParallel() {
	scheduler := newFSMApplyScheduler(r)

	saturation := newSaturationMetric([]string{"raft", "thread", "fsm", "saturation"}, 1*time.Second)

	for {
		saturation.sleeping()

		select {
		case ptr := <-r.fsmMutateCh:
			saturation.working()

			switch req := ptr.(type) {
			case []*commitTuple:
				for _, ct := range req {
					scheduler.admit(ct)
				}

			case *restoreFuture:
				scheduler.restore(req)

			default:
				panic(fmt.Errorf("bad type passed to fsmMutateCh: %#v", ptr))
			}

		case req := <-r.fsmSnapshotCh:
			saturation.working()

			scheduler.snapshot(req)

		case <-r.shutdownCh:
			return
		}
	}
}
