// Copyright IBM Corp. 2026
// SPDX-License-Identifier: MPL-2.0

package raft

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/hashicorp/go-msgpack/v2/codec"
)

// errDepTestApply is returned by depTestFSM applies that are asked to fail.
var errDepTestApply = errors.New("dep test apply failure")

// depTestCmd is the JSON payload used by the parallel apply tests.
type depTestCmd struct {
	// Kind selects the declared dependency:
	//   "indep": no dependency (0)
	//   "prev":  depends on the immediately preceding index
	//   "fail":  no dependency, but Apply returns an error
	Kind string `json:"k"`

	// Gate, when non-empty, blocks Apply on a named gate while the test
	// drives scheduling.
	Gate string `json:"g,omitempty"`

	// Data is recorded into the FSM state.
	Data string `json:"d,omitempty"`
}

// depTestGate lets a test hold an FSM.Apply (or StoreConfiguration) call
// open and observe when it starts.
type depTestGate struct {
	startedCh chan struct{}
	releaseCh chan struct{}

	startOnce sync.Once
	endOnce   sync.Once
}

func newDepTestGate() *depTestGate {
	return &depTestGate{
		startedCh: make(chan struct{}),
		releaseCh: make(chan struct{}),
	}
}

func (g *depTestGate) signalStarted() {
	g.startOnce.Do(func() { close(g.startedCh) })
}

func (g *depTestGate) release() {
	g.endOnce.Do(func() { close(g.releaseCh) })
}

func (g *depTestGate) waitStarted(t *testing.T, timeout time.Duration) {
	t.Helper()
	select {
	case <-g.startedCh:
	case <-time.After(timeout):
		t.Fatalf("timed out waiting for gated operation to start")
	}
}

func (g *depTestGate) assertNotStarted(t *testing.T, within time.Duration) {
	t.Helper()
	select {
	case <-g.startedCh:
		t.Fatalf("gated operation started unexpectedly")
	case <-time.After(within):
	}
}

// depTestGates is the set of gates shared with an FSM (or all FSMs of a
// cluster).
type depTestGates struct {
	sync.Mutex
	m map[string]*depTestGate
}

func newDepTestGates() *depTestGates {
	return &depTestGates{m: make(map[string]*depTestGate)}
}

func (g *depTestGates) get(name string) *depTestGate {
	g.Lock()
	defer g.Unlock()
	gate, ok := g.m[name]
	if !ok {
		gate = newDepTestGate()
		g.m[name] = gate
	}
	return gate
}

func (g *depTestGates) lookup(name string) *depTestGate {
	g.Lock()
	defer g.Unlock()
	return g.m[name]
}

func (g *depTestGates) releaseAll() {
	g.Lock()
	defer g.Unlock()
	for _, gate := range g.m {
		gate.release()
	}
}

type depTestSnapshotObs struct {
	dataLen  int
	inFlight int32
}

// depTestFSM implements DependencyFSM (and optionally ConfigurationStore)
// for the parallel scheduling tests.
type depTestFSM struct {
	t     *testing.T
	gates *depTestGates

	// configGate names a gate that StoreConfiguration blocks on; empty
	// disables gating.
	configGate string

	stateMu sync.Mutex
	data    [][]byte

	// Monotonic event counter used to assert apply ordering without
	// relying on clock granularity.
	eventSeq uint64

	// Per-index observations.
	startSeq  map[uint64]uint64
	finishSeq map[uint64]uint64
	applied   map[uint64]bool
	started   map[uint64]bool
	configs   []uint64

	snapshots []depTestSnapshotObs

	inFlight    atomic.Int32
	maxInFlight atomic.Int32

	configOverlap atomic.Bool
}

func newDepTestFSM(t *testing.T, gates *depTestGates, configGate string) *depTestFSM {
	if gates == nil {
		gates = newDepTestGates()
	}
	return &depTestFSM{
		t:         t,
		gates:     gates,
		configGate: configGate,
		startSeq:  make(map[uint64]uint64),
		finishSeq: make(map[uint64]uint64),
		applied:   make(map[uint64]bool),
		started:   make(map[uint64]bool),
	}
}

// withoutConfigurationStore hides the ConfigurationStore implementation so
// the FSM only satisfies DependencyFSM (not ConfigurationStore).
func (f *depTestFSM) withoutConfigurationStore() FSM {
	return &depTestFSMNoConfig{FSM: f}
}

// depTestFSMNoConfig embeds the FSM interface, so the promoted method set is
// exactly FSM; ApplyDependency is provided explicitly.
type depTestFSMNoConfig struct {
	FSM
}

func (f *depTestFSMNoConfig) ApplyDependency(log *Log) uint64 {
	return f.FSM.(DependencyFSM).ApplyDependency(log)
}

func (f *depTestFSM) decode(log *Log) depTestCmd {
	var cmd depTestCmd
	if len(log.Data) > 0 {
		if err := json.Unmarshal(log.Data, &cmd); err != nil {
			f.t.Fatalf("failed to decode test command: %v", err)
		}
	}
	return cmd
}

// ApplyDependency implements DependencyFSM.
func (f *depTestFSM) ApplyDependency(log *Log) uint64 {
	switch f.decode(log).Kind {
	case "prev":
		return log.Index - 1
	default:
		return 0
	}
}

// Apply implements FSM.
func (f *depTestFSM) Apply(log *Log) interface{} {
	cmd := f.decode(log)

	inFlight := f.inFlight.Add(1)
	for {
		max := f.maxInFlight.Load()
		if inFlight <= max || f.maxInFlight.CompareAndSwap(max, inFlight) {
			break
		}
	}
	defer f.inFlight.Add(-1)

	f.stateMu.Lock()
	f.eventSeq++
	f.started[log.Index] = true
	f.startSeq[log.Index] = f.eventSeq
	f.stateMu.Unlock()

	if cmd.Gate != "" {
		gate := f.gates.get(cmd.Gate)
		gate.signalStarted()
		<-gate.releaseCh
	}

	var resp interface{}
	if cmd.Kind == "fail" {
		resp = errDepTestApply
	} else {
		f.stateMu.Lock()
		f.data = append(f.data, log.Data)
		f.stateMu.Unlock()
		resp = cmd.Data
	}

	f.stateMu.Lock()
	f.eventSeq++
	f.applied[log.Index] = true
	f.finishSeq[log.Index] = f.eventSeq
	f.stateMu.Unlock()

	return resp
}

// Snapshot implements FSM.
func (f *depTestFSM) Snapshot() (FSMSnapshot, error) {
	inFlight := f.inFlight.Load()

	f.stateMu.Lock()
	data := make([][]byte, len(f.data))
	copy(data, f.data)
	f.snapshots = append(f.snapshots, depTestSnapshotObs{
		dataLen:  len(data),
		inFlight: inFlight,
	})
	f.stateMu.Unlock()

	return &depTestSnapshot{data: data}, nil
}

// Restore implements FSM.
func (f *depTestFSM) Restore(inp io.ReadCloser) error {
	defer func() { _ = inp.Close() }()
	hd := codec.MsgpackHandle{}
	dec := codec.NewDecoder(inp, &hd)

	var data [][]byte
	if err := dec.Decode(&data); err != nil {
		return err
	}

	f.stateMu.Lock()
	f.data = data
	f.stateMu.Unlock()
	return nil
}

func (f *depTestFSM) StoreConfiguration(index uint64, _ Configuration) {
	if f.inFlight.Load() != 0 {
		f.configOverlap.Store(true)
	}
	if f.configGate != "" {
		if gate := f.gates.lookup(f.configGate); gate != nil {
			gate.signalStarted()
			<-gate.releaseCh
		}
	}
	if f.inFlight.Load() != 0 {
		f.configOverlap.Store(true)
	}

	f.stateMu.Lock()
	f.configs = append(f.configs, index)
	f.stateMu.Unlock()
}

func (f *depTestFSM) wasApplied(index uint64) bool {
	f.stateMu.Lock()
	defer f.stateMu.Unlock()
	return f.applied[index]
}

func (f *depTestFSM) numStarted() int {
	f.stateMu.Lock()
	defer f.stateMu.Unlock()
	return len(f.started)
}

func (f *depTestFSM) startFinish(index uint64) (uint64, uint64) {
	f.stateMu.Lock()
	defer f.stateMu.Unlock()
	return f.startSeq[index], f.finishSeq[index]
}

func (f *depTestFSM) dataCopy() [][]byte {
	f.stateMu.Lock()
	defer f.stateMu.Unlock()
	out := make([][]byte, len(f.data))
	copy(out, f.data)
	return out
}

func (f *depTestFSM) configsCopy() []uint64 {
	f.stateMu.Lock()
	defer f.stateMu.Unlock()
	out := make([]uint64, len(f.configs))
	copy(out, f.configs)
	return out
}

func (f *depTestFSM) lastSnapshotObs() depTestSnapshotObs {
	f.stateMu.Lock()
	defer f.stateMu.Unlock()
	return f.snapshots[len(f.snapshots)-1]
}

func (f *depTestFSM) numSnapshots() int {
	f.stateMu.Lock()
	defer f.stateMu.Unlock()
	return len(f.snapshots)
}

type depTestSnapshot struct {
	data [][]byte
}

func (s *depTestSnapshot) Persist(sink SnapshotSink) error {
	hd := codec.MsgpackHandle{}
	enc := codec.NewEncoder(sink, &hd)
	if err := enc.Encode(s.data); err != nil {
		_ = sink.Cancel()
		return err
	}
	return sink.Close()
}

func (s *depTestSnapshot) Release() {}

// makeDepCluster builds an n-node cluster running depTestFSM instances.
func makeDepCluster(t *testing.T, n int, makeFSM func() FSM) *cluster {
	t.Helper()

	conf := inmemConfig(t)
	// Keep automatic snapshots out of the tests.
	conf.SnapshotThreshold = 100000
	conf.TrailingLogs = 10

	c, err := makeCluster(t, &MakeClusterOpts{
		Peers:       n,
		Bootstrap:   true,
		Conf:        conf,
		MakeFSMFunc: makeFSM,
	})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	return c
}

func depApply(t *testing.T, r *Raft, cmd depTestCmd) ApplyFuture {
	t.Helper()
	buf, err := json.Marshal(cmd)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	return r.Apply(buf, 5*time.Second)
}

func futureErrCh(f Future) <-chan error {
	ch := make(chan error, 1)
	go func() { ch <- f.Error() }()
	return ch
}

// TestDepFSM_IndependentAppliesRunConcurrently verifies that entries that
// declare no dependency really overlap inside FSM.Apply.
func TestDepFSM_IndependentAppliesRunConcurrently(t *testing.T) {
	gates := newDepTestGates()
	fsm := newDepTestFSM(t, gates, "")
	c := makeDepCluster(t, 1, func() FSM { return fsm })
	defer c.Close()
	defer gates.releaseAll()

	leader := c.Leader()
	gateA := gates.get("A")
	gateB := gates.get("B")

	fa := depApply(t, leader, depTestCmd{Kind: "indep", Gate: "A", Data: "a"})
	fb := depApply(t, leader, depTestCmd{Kind: "indep", Gate: "B", Data: "b"})

	gateA.waitStarted(t, 5*time.Second)
	gateB.waitStarted(t, 5*time.Second)

	if max := fsm.maxInFlight.Load(); max < 2 {
		t.Fatalf("expected concurrent applies, max observed in flight: %d", max)
	}

	gateA.release()
	gateB.release()

	if err := fa.Error(); err != nil {
		t.Fatalf("apply A err: %v", err)
	}
	if err := fb.Error(); err != nil {
		t.Fatalf("apply B err: %v", err)
	}
	if !fsm.wasApplied(fa.Index()) || !fsm.wasApplied(fb.Index()) {
		t.Fatalf("both applies expected to run")
	}
}

// TestDepFSM_DependencyIsRespected verifies that a dependent entry only
// starts once every entry in its declared dependency prefix has finished.
func TestDepFSM_DependencyIsRespected(t *testing.T) {
	gates := newDepTestGates()
	fsm := newDepTestFSM(t, gates, "")
	c := makeDepCluster(t, 1, func() FSM { return fsm })
	defer c.Close()
	defer gates.releaseAll()

	leader := c.Leader()
	gate := gates.get("first")

	fa := depApply(t, leader, depTestCmd{Kind: "indep", Gate: "first", Data: "a"})
	gate.waitStarted(t, 5*time.Second)

	fb := depApply(t, leader, depTestCmd{Kind: "prev", Data: "b"})

	// The dependent entry must not start while the first is gated open.
	time.Sleep(200 * time.Millisecond)
	if fsm.numStarted() != 1 {
		t.Fatalf("dependent apply started before its dependency finished")
	}

	gate.release()

	if err := fb.Error(); err != nil {
		t.Fatalf("dependent apply err: %v", err)
	}
	if err := fa.Error(); err != nil {
		t.Fatalf("first apply err: %v", err)
	}

	startB, _ := fsm.startFinish(fb.Index())
	_, finishA := fsm.startFinish(fa.Index())

	if startB <= finishA {
		t.Fatalf("dependent apply at %d (start seq %d) did not wait for dependency at %d (finish seq %d)",
			fb.Index(), startB, fa.Index(), finishA)
	}
}

// TestDepFSM_FailedDependencyAbortsDependents verifies error propagation:
// a failed apply aborts later entries that depend on it, while independent
// entries still run; no future is left hanging.
func TestDepFSM_FailedDependencyAbortsDependents(t *testing.T) {
	var fsms []*depTestFSM
	c := makeDepCluster(t, 3, func() FSM {
		f := newDepTestFSM(t, newDepTestGates(), "")
		fsms = append(fsms, f)
		return f
	})
	defer c.Close()
	defer func() {
		for _, f := range fsms {
			f.gates.releaseAll()
		}
	}()

	leader := c.Leader()

	fa := depApply(t, leader, depTestCmd{Kind: "fail", Data: "fail"})
	fb := depApply(t, leader, depTestCmd{Kind: "prev", Data: "dependent"})
	fc := depApply(t, leader, depTestCmd{Kind: "indep", Data: "independent"})

	if err := fa.Error(); err != nil {
		t.Fatalf("failed apply future err should be nil, got: %v", err)
	}
	resp, ok := fa.Response().(error)
	if !ok || !errors.Is(resp, errDepTestApply) {
		t.Fatalf("expected apply error response, got: %#v", fa.Response())
	}

	if err := fb.Error(); !errors.Is(err, ErrDependencyAborted) {
		t.Fatalf("expected ErrDependencyAborted for dependent entry, got: %v", err)
	}

	if err := fc.Error(); err != nil {
		t.Fatalf("independent apply should still run, got err: %v", err)
	}
	if fc.Response() != "independent" {
		t.Fatalf("independent apply should have its normal response, got: %#v", fc.Response())
	}

	// Every cluster member must have made the same scheduling decision.
	// Followers apply asynchronously via replication, so poll for them.
	leaderIdxA, leaderIdxC := fa.Index(), fc.Index()
	leaderIdxB := fb.Index()
	deadline := time.Now().Add(5 * time.Second)
	for i, f := range fsms {
		for {
			appliedA := f.wasApplied(leaderIdxA)
			abortedB := !f.wasApplied(leaderIdxB)
			appliedC := f.wasApplied(leaderIdxC)
			if appliedA && abortedB && appliedC {
				break
			}
			if time.Now().After(deadline) {
				t.Fatalf("node %d scheduling mismatch: failed applied=%v dependent skipped=%v independent applied=%v",
					i, appliedA, abortedB, appliedC)
			}
			time.Sleep(10 * time.Millisecond)
		}
	}
}

// TestDepFSM_BarrierDrainsAfterFailure verifies barriers still complete
// after an aborted prefix and that normal applies continue afterwards.
func TestDepFSM_BarrierDrainsAfterFailure(t *testing.T) {
	gates := newDepTestGates()
	fsm := newDepTestFSM(t, gates, "")
	c := makeDepCluster(t, 1, func() FSM { return fsm })
	defer c.Close()
	defer gates.releaseAll()

	leader := c.Leader()

	fa := depApply(t, leader, depTestCmd{Kind: "fail", Data: "fail"})
	fb := depApply(t, leader, depTestCmd{Kind: "prev", Data: "dependent"})

	if err := fa.Error(); err != nil {
		t.Fatalf("err: %v", err)
	}
	if err := fb.Error(); !errors.Is(err, ErrDependencyAborted) {
		t.Fatalf("expected abort, got: %v", err)
	}

	if err := leader.Barrier(5 * time.Second).Error(); err != nil {
		t.Fatalf("barrier err: %v", err)
	}

	fc := depApply(t, leader, depTestCmd{Kind: "indep", Data: "after"})
	if err := fc.Error(); err != nil {
		t.Fatalf("post-barrier apply err: %v", err)
	}
	if !fsm.wasApplied(fc.Index()) {
		t.Fatalf("post-barrier entry should have been applied")
	}
}

// TestDepFSM_ConfigurationChangesAreSerialized verifies that membership
// changes never overlap applies and that applies following a config change
// wait for the configuration to be stored.
func TestDepFSM_ConfigurationChangesAreSerialized(t *testing.T) {
	gates := newDepTestGates()
	fsm := newDepTestFSM(t, gates, "config")
	c := makeDepCluster(t, 1, func() FSM { return fsm })
	defer c.Close()
	defer gates.releaseAll()

	leader := c.Leader()

	// Flush the bootstrap configuration entry, which is also handed to
	// StoreConfiguration, before observing the baseline.
	if err := leader.Barrier(5 * time.Second).Error(); err != nil {
		t.Fatalf("barrier err: %v", err)
	}
	baseConfigs := len(fsm.configsCopy())

	// Gate an independent apply open.
	appGate := gates.get("app")
	fa := depApply(t, leader, depTestCmd{Kind: "indep", Gate: "app", Data: "a"})
	appGate.waitStarted(t, 5*time.Second)

	// A membership change committed while the apply is open must not run.
	configGate := gates.get("config")
	fcfgCh := futureErrCh(leader.AddNonvoter(
		ServerID("dep-test-nonvoter"), ServerAddress("9.9.9.9:1234"), 0, 5*time.Second))
	select {
	case err := <-fcfgCh:
		t.Fatalf("config change completed while apply was in flight: %v", err)
	case <-time.After(300 * time.Millisecond):
	}
	if len(fsm.configsCopy()) != baseConfigs {
		t.Fatalf("StoreConfiguration ran while an apply was in flight")
	}

	// Finishing the apply lets the config change proceed, which is itself
	// gated open.
	appGate.release()
	configGate.waitStarted(t, 5*time.Second)

	if v := fsm.inFlight.Load(); v != 0 {
		t.Fatalf("StoreConfiguration started with %d applies in flight", v)
	}

	// An entry committed after the config entry must not apply until the
	// configuration has been stored.
	afterGate := gates.get("after")
	fAfter := depApply(t, leader, depTestCmd{Kind: "indep", Gate: "after", Data: "after"})
	afterGate.assertNotStarted(t, 300*time.Millisecond)

	configGate.release()

	afterGate.waitStarted(t, 5*time.Second)
	afterGate.release()
	if err := fAfter.Error(); err != nil {
		t.Fatalf("post-config apply err: %v", err)
	}
	if err := fa.Error(); err != nil {
		t.Fatalf("gated apply err: %v", err)
	}

	select {
	case err := <-fcfgCh:
		if err != nil {
			t.Fatalf("AddNonvoter err: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatalf("AddNonvoter did not complete")
	}

	// Remove the non-voter; this second config entry must also be
	// serialized.
	if err := leader.RemoveServer(ServerID("dep-test-nonvoter"), 0, 5*time.Second).Error(); err != nil {
		t.Fatalf("RemoveServer err: %v", err)
	}

	if fsm.configOverlap.Load() {
		t.Fatalf("StoreConfiguration overlapped an FSM.Apply")
	}
	if got := len(fsm.configsCopy()); got != baseConfigs+2 {
		t.Fatalf("expected %d stored configurations, got: %v", baseConfigs+2, got)
	}
}

// TestDepFSM_SnapshotIsConsistent verifies Snapshot waits for all in-flight
// applies and observes a quiescent, complete FSM.
func TestDepFSM_SnapshotIsConsistent(t *testing.T) {
	gates := newDepTestGates()
	fsm := newDepTestFSM(t, gates, "")
	c := makeDepCluster(t, 1, func() FSM { return fsm })
	defer c.Close()
	defer gates.releaseAll()

	leader := c.Leader()

	snapGate := gates.get("snap")
	fa := depApply(t, leader, depTestCmd{Kind: "indep", Gate: "snap", Data: "gated"})
	snapGate.waitStarted(t, 5*time.Second)

	fb := depApply(t, leader, depTestCmd{Kind: "indep", Data: "b"})
	fc := depApply(t, leader, depTestCmd{Kind: "indep", Data: "c"})
	if err := fb.Error(); err != nil {
		t.Fatalf("err: %v", err)
	}
	if err := fc.Error(); err != nil {
		t.Fatalf("err: %v", err)
	}

	snapCh := futureErrCh(leader.Snapshot())
	select {
	case err := <-snapCh:
		t.Fatalf("snapshot completed while an apply was in flight: %v", err)
	case <-time.After(300 * time.Millisecond):
	}

	snapGate.release()
	if err := fa.Error(); err != nil {
		t.Fatalf("gated apply err: %v", err)
	}

	select {
	case err := <-snapCh:
		if err != nil && !errors.Is(err, ErrNothingNewToSnapshot) {
			t.Fatalf("snapshot err: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatalf("snapshot did not complete")
	}

	if fsm.numSnapshots() == 0 {
		t.Fatalf("expected at least one snapshot")
	}
	obs := fsm.lastSnapshotObs()
	if obs.inFlight != 0 {
		t.Fatalf("snapshot observed %d applies in flight", obs.inFlight)
	}
	if obs.dataLen != 3 {
		t.Fatalf("snapshot should contain all 3 entries, got %d", obs.dataLen)
	}
}

// TestDepFSM_ConfigWithoutConfigStoreStillSerializes exercises a
// DependencyFSM that does not implement ConfigurationStore: configuration
// entries stay serial and snapshots still work.
func TestDepFSM_ConfigWithoutConfigStoreStillSerializes(t *testing.T) {
	gates := newDepTestGates()
	fsm := newDepTestFSM(t, gates, "")
	if _, ok := FSM(fsm.withoutConfigurationStore()).(ConfigurationStore); ok {
		t.Fatalf("test wrapper should not implement ConfigurationStore")
	}
	c := makeDepCluster(t, 1, func() FSM { return fsm.withoutConfigurationStore() })
	defer c.Close()
	defer gates.releaseAll()

	leader := c.Leader()

	if err := leader.AddNonvoter(
		ServerID("dep-test-nonvoter"), ServerAddress("9.9.9.9:1234"), 0, 5*time.Second).Error(); err != nil {
		t.Fatalf("AddNonvoter err: %v", err)
	}
	if err := leader.RemoveServer(ServerID("dep-test-nonvoter"), 0, 5*time.Second).Error(); err != nil {
		t.Fatalf("RemoveServer err: %v", err)
	}

	f := depApply(t, leader, depTestCmd{Kind: "indep", Data: "after-config"})
	if err := f.Error(); err != nil {
		t.Fatalf("apply err: %v", err)
	}

	if err := leader.Snapshot().Error(); err != nil {
		t.Fatalf("snapshot err: %v", err)
	}
}

// TestDepFSM_RestoreCommittedLogs verifies the scheduler handles committed
// entries replayed at startup (before the FSM goroutine exists) the same as
// live entries, including gap rollover and dependency waits.
func TestDepFSM_RestoreCommittedLogs(t *testing.T) {
	conf := inmemConfig(t)
	conf.RestoreCommittedLogs = true

	gates := newDepTestGates()
	fsm := newDepTestFSM(t, gates, "")
	opts := &MakeClusterOpts{
		Peers:              1,
		Bootstrap:          true,
		Conf:               conf,
		CommitTrackingLogs: true,
		MakeFSMFunc: func() FSM {
			return fsm
		},
	}
	c, err := makeCluster(t, opts)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	defer c.Close()
	defer gates.releaseAll()

	leader := c.Leader()

	// Commit a first batch and snapshot it, so the configuration survives
	// the restart through the snapshot (matching the library's own
	// RestoreCommittedLogs test).
	var future ApplyFuture
	for i := 0; i < 30; i++ {
		future = depApply(t, leader, depTestCmd{Kind: "indep", Data: fmt.Sprintf("d%d", i)})
	}
	if err := future.Error(); err != nil {
		t.Fatalf("apply err: %v", err)
	}
	if err := leader.Snapshot().Error(); err != nil {
		t.Fatalf("snapshot err: %v", err)
	}

	// A second batch mixing chain-dependent entries that replay has to
	// honor. They sit after the snapshot.
	for i := 30; i < 40; i++ {
		kind := "indep"
		if i%2 == 1 {
			kind = "prev"
		}
		future = depApply(t, leader, depTestCmd{Kind: kind, Data: fmt.Sprintf("d%d", i)})
	}
	if err := future.Error(); err != nil {
		t.Fatalf("apply err: %v", err)
	}

	// Restart the node reusing its stores and FSM.
	if err := leader.Shutdown().Error(); err != nil {
		t.Fatalf("shutdown err: %v", err)
	}
	_, trans2 := NewInmemTransport(leader.trans.LocalAddr())
	cfg := leader.config()
	r, err := NewRaft(&cfg, fsm, leader.logs, leader.stable, leader.snapshots, trans2)
	if err != nil {
		t.Fatalf("restart err: %v", err)
	}
	c.rafts[0] = r

	// Startup restores the FSM from the snapshot (first 30 entries), then
	// replays from the staged commit index; entries past that point are
	// re-delivered once the restarted node elects itself. Eventually the
	// FSM must contain all 40 committed entries exactly once.
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if len(fsm.dataCopy()) == 40 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if got := len(fsm.dataCopy()); got != 40 {
		t.Fatalf("expected FSM length 40 after replay/recommit, got %d", got)
	}

	// The scheduler must be fully functional after replay: once leader
	// again, a barrier and a new dependent entry complete without hanging
	// on gap frontiers.
	deadline = time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if err := r.Barrier(5 * time.Second).Error(); err == nil {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if err := r.Barrier(0).Error(); err != nil {
		t.Fatalf("post-restart barrier err: %v", err)
	}
	f := depApply(t, r, depTestCmd{Kind: "prev", Data: "after-restart"})
	if err := f.Error(); err != nil {
		t.Fatalf("post-restart apply err: %v", err)
	}
	if got := len(fsm.dataCopy()); got != 41 {
		t.Fatalf("expected FSM length 41 after post-restart apply, got %d", got)
	}
}

// TestDepFSM_SnapshotInstallRestore exercises the scheduler's restore path
// when a follower is brought up via InstallSnapshot.
func TestDepFSM_SnapshotInstallRestore(t *testing.T) {
	var fsms []*depTestFSM
	c := makeDepCluster(t, 3, func() FSM {
		f := newDepTestFSM(t, newDepTestGates(), "")
		fsms = append(fsms, f)
		return f
	})
	defer c.Close()
	defer func() {
		for _, f := range fsms {
			f.gates.releaseAll()
		}
	}()

	leader := c.Leader()
	behind := c.Followers()[0]
	c.Disconnect(behind.localAddr)

	var future Future
	for i := 0; i < 100; i++ {
		future = depApply(t, leader, depTestCmd{Kind: "indep", Data: fmt.Sprintf("test%d", i)})
	}
	if err := future.Error(); err != nil {
		t.Fatalf("err: %v", err)
	}

	for _, r := range c.rafts {
		if err := r.Snapshot().Error(); err != nil && !errors.Is(err, ErrNothingNewToSnapshot) {
			t.Fatalf("snapshot err: %v", err)
		}
	}

	c.FullyConnect()

	want := fsms[c.IndexOf(leader)].dataCopy()
	limit := time.Now().Add(10 * time.Second)
	for {
		got := fsms[c.IndexOf(behind)].dataCopy()
		equal := len(got) == len(want)
		if equal {
			for i := range want {
				if !bytes.Equal(want[i], got[i]) {
					equal = false
					break
				}
			}
		}
		if equal {
			return
		}
		if time.Now().After(limit) {
			t.Fatalf("follower did not converge after install snapshot: want %d entries, got %d",
				len(want), len(got))
		}
		time.Sleep(20 * time.Millisecond)
	}
}
