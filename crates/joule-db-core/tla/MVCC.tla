-------------------------------- MODULE MVCC --------------------------------
(***************************************************************************)
(* TLA+ specification of Multi-Version Concurrency Control (MVCC) with     *)
(* Snapshot Isolation as implemented in JouleDB                            *)
(* (joule-db-core/src/concurrency/mvcc.rs).                                *)
(*                                                                         *)
(* Models:                                                                 *)
(*   - Version chains per key                                              *)
(*   - read_ts assigned at transaction start (immutable)                   *)
(*   - commit_ts assigned at commit time from monotonic oracle             *)
(*   - First-Committer-Wins conflict detection                             *)
(*   - Visibility: write_ts <= read_ts AND committed                       *)
(*                                                                         *)
(* Model parameters:                                                       *)
(*   Tx   = {tx1, tx2, tx3}   (3 transactions)                            *)
(*   Key  = {k1, k2}          (2 keys)                                     *)
(*   MaxTS = 3                (max timestamp value)                        *)
(***************************************************************************)

EXTENDS Naturals, FiniteSets, Sequences, TLC

CONSTANTS Tx, Key, MaxTS

VARIABLES
    txState,       \* txState[t] \in {"idle", "active", "committed", "aborted"}
    readTS,        \* readTS[t]: read timestamp assigned at begin (0 if idle)
    commitTS,      \* commitTS[t]: commit timestamp (0 if not yet committed)
    writeSet,      \* writeSet[t]: set of keys written by transaction t
    readSet,       \* readSet[t]: set of keys read by transaction t
    versions,      \* versions[k]: sequence of version records for key k
                   \*   each: [writer |-> tx, writeTs |-> ts, committed |-> BOOLEAN, val |-> v]
    tsOracle,      \* global monotonic timestamp counter
    appliedLog     \* appliedLog[t]: set of {[key |-> k, val |-> v]} applied by t
                   \*   (used for state machine safety / read consistency checking)

vars == <<txState, readTS, commitTS, writeSet, readSet, versions, tsOracle, appliedLog>>

Nil == "Nil"

(***************************************************************************)
(* Helper operators                                                        *)
(***************************************************************************)

\* Allocate next timestamp from oracle
NextTS == tsOracle + 1

\* Find the visible version for transaction t reading key k:
\*   - committed = TRUE
\*   - writeTs <= readTS[t]
\*   - newest such version (highest writeTs)
VisibleVersion(k, t) ==
    LET validVersions == {i \in 1..Len(versions[k]) :
                            /\ versions[k][i].committed = TRUE
                            /\ versions[k][i].writeTs <= readTS[t]}
    IN IF validVersions = {} THEN Nil
       ELSE LET maxIdx == CHOOSE i \in validVersions :
                    \A j \in validVersions : versions[k][j].writeTs <= versions[k][i].writeTs
            IN versions[k][maxIdx]

\* Find version written by transaction t for key k (for read-your-writes)
OwnVersion(k, t) ==
    LET ownVersions == {i \in 1..Len(versions[k]) : versions[k][i].writer = t}
    IN IF ownVersions = {} THEN Nil
       ELSE versions[k][CHOOSE i \in ownVersions : TRUE]

\* Check if another uncommitted transaction holds a write on key k
\* (First-Committer-Wins: only one uncommitted writer at a time)
HasWriteConflict(k, t) ==
    \E i \in 1..Len(versions[k]) :
        /\ versions[k][i].writer /= t
        /\ versions[k][i].committed = FALSE

\* Check if a committed version exists with writeTs > readTS[t]
\* (write-write conflict at validation/commit time)
HasCommittedConflict(k, t) ==
    \E i \in 1..Len(versions[k]) :
        /\ versions[k][i].committed = TRUE
        /\ versions[k][i].writeTs > readTS[t]

(***************************************************************************)
(* Actions                                                                 *)
(***************************************************************************)

\* Begin a new transaction: assigns read_ts from oracle
\* Corresponds to MvccTransaction::new() in mvcc.rs
BeginTx(t) ==
    /\ txState[t] = "idle"
    /\ tsOracle < MaxTS
    /\ LET ts == NextTS IN
       /\ tsOracle' = ts
       /\ txState' = [txState EXCEPT ![t] = "active"]
       /\ readTS' = [readTS EXCEPT ![t] = ts]
       /\ commitTS' = [commitTS EXCEPT ![t] = 0]
       /\ writeSet' = [writeSet EXCEPT ![t] = {}]
       /\ readSet' = [readSet EXCEPT ![t] = {}]
       /\ appliedLog' = [appliedLog EXCEPT ![t] = {}]
       /\ UNCHANGED versions

\* Transaction t reads key k
\* Corresponds to MvccTransaction::get() in mvcc.rs
\*   - First checks own write set (read-your-writes)
\*   - Then finds visible committed version (write_ts <= read_ts)
Read(t, k) ==
    /\ txState[t] = "active"
    /\ readSet' = [readSet EXCEPT ![t] = readSet[t] \cup {k}]
    /\ LET own == OwnVersion(k, t)
           vis == VisibleVersion(k, t)
           result == IF own /= Nil THEN own.val
                     ELSE IF vis /= Nil THEN vis.val
                     ELSE Nil
       IN appliedLog' = [appliedLog EXCEPT ![t] = appliedLog[t] \cup
                            {[key |-> k, val |-> result]}]
    /\ UNCHANGED <<txState, readTS, commitTS, writeSet, versions, tsOracle>>

\* Transaction t writes key k with abstract value v
\* Corresponds to MvccTransaction::put() in mvcc.rs
\*   - Checks for write-write conflict (another uncommitted writer)
\*   - If conflict, transaction aborts (First-Committer-Wins)
\*   - Otherwise, creates uncommitted version (write intent)
Write(t, k, v) ==
    /\ txState[t] = "active"
    /\ v \in 1..MaxTS  \* abstract value domain
    /\ IF HasWriteConflict(k, t)
       THEN
         \* Abort on write-write conflict (FCW)
         /\ txState' = [txState EXCEPT ![t] = "aborted"]
         \* Remove our uncommitted versions on abort
         /\ versions' = [kk \in Key |->
              SelectSeq(versions[kk],
                LAMBDA ver : ~(ver.writer = t /\ ver.committed = FALSE))]
         /\ UNCHANGED <<readTS, commitTS, writeSet, readSet, tsOracle, appliedLog>>
       ELSE
         \* Write intent: create uncommitted version
         /\ LET newVer == [writer    |-> t,
                           writeTs   |-> readTS[t],  \* provisional; updated at commit
                           committed |-> FALSE,
                           val       |-> v]
            IN
            \* If we already wrote this key, update in place
            IF k \in writeSet[t]
            THEN versions' = [versions EXCEPT ![k] =
                    [i \in 1..Len(versions[k]) |->
                        IF versions[k][i].writer = t /\ versions[k][i].committed = FALSE
                        THEN [versions[k][i] EXCEPT !.val = v]
                        ELSE versions[k][i]]]
            ELSE versions' = [versions EXCEPT ![k] = Append(versions[k], newVer)]
         /\ writeSet' = [writeSet EXCEPT ![t] = writeSet[t] \cup {k}]
         /\ txState' = txState
         /\ UNCHANGED <<readTS, commitTS, readSet, tsOracle, appliedLog>>

\* Transaction t commits
\* Corresponds to MvccTransaction::commit() in mvcc.rs
\*   - Validates: no committed version with writeTs > readTS for written keys
\*   - Assigns commit_ts from oracle
\*   - Marks all write intents as committed with commit_ts
Commit(t) ==
    /\ txState[t] = "active"
    /\ tsOracle < MaxTS
    \* Validation: check for serialization conflicts (first-committer-wins)
    /\ \A k \in writeSet[t] : ~HasCommittedConflict(k, t)
    /\ LET cts == NextTS IN
       /\ tsOracle' = cts
       /\ commitTS' = [commitTS EXCEPT ![t] = cts]
       /\ txState' = [txState EXCEPT ![t] = "committed"]
       \* Update all write intents: set writeTs to commit_ts, mark committed
       /\ versions' = [k \in Key |->
            [i \in 1..Len(versions[k]) |->
                IF versions[k][i].writer = t /\ versions[k][i].committed = FALSE
                THEN [versions[k][i] EXCEPT !.writeTs = cts,
                                            !.committed = TRUE]
                ELSE versions[k][i]]]
       /\ UNCHANGED <<readTS, writeSet, readSet, appliedLog>>

\* Transaction t aborts (voluntary or due to conflict)
\* Corresponds to MvccTransaction::rollback() in mvcc.rs
Abort(t) ==
    /\ txState[t] = "active"
    /\ txState' = [txState EXCEPT ![t] = "aborted"]
    \* Remove all uncommitted versions from this transaction
    /\ versions' = [k \in Key |->
         SelectSeq(versions[k],
           LAMBDA ver : ~(ver.writer = t /\ ver.committed = FALSE))]
    /\ UNCHANGED <<readTS, commitTS, writeSet, readSet, tsOracle, appliedLog>>

(***************************************************************************)
(* Initial state                                                           *)
(***************************************************************************)
Init ==
    /\ txState    = [t \in Tx |-> "idle"]
    /\ readTS     = [t \in Tx |-> 0]
    /\ commitTS   = [t \in Tx |-> 0]
    /\ writeSet   = [t \in Tx |-> {}]
    /\ readSet    = [t \in Tx |-> {}]
    /\ versions   = [k \in Key |-> <<>>]
    /\ tsOracle   = 0
    /\ appliedLog = [t \in Tx |-> {}]

(***************************************************************************)
(* Next-state relation                                                     *)
(***************************************************************************)
Next ==
    \/ \E t \in Tx : BeginTx(t)
    \/ \E t \in Tx, k \in Key : Read(t, k)
    \/ \E t \in Tx, k \in Key, v \in 1..MaxTS : Write(t, k, v)
    \/ \E t \in Tx : Commit(t)
    \/ \E t \in Tx : Abort(t)

Spec == Init /\ [][Next]_vars

(***************************************************************************)
(* Safety Properties                                                       *)
(***************************************************************************)

\* 1. No Dirty Reads: a transaction never reads uncommitted data from
\*    another transaction.
\*    Formally: any value read by t via VisibleVersion must come from a
\*    committed version (by construction of VisibleVersion which requires
\*    committed = TRUE). We verify that no read result in appliedLog
\*    matches an uncommitted foreign version's value at the time of reading.
NoDirtyReads ==
    \A t \in Tx :
        txState[t] \in {"active", "committed"} =>
            \A entry \in appliedLog[t] :
                entry.val /= Nil =>
                    \* The value either came from t's own write or from a committed version
                    \/ OwnVersion(entry.key, t) /= Nil
                    \/ \E i \in 1..Len(versions[entry.key]) :
                        /\ versions[entry.key][i].committed = TRUE
                        /\ versions[entry.key][i].val = entry.val

\* 2. No Lost Updates: write-write conflicts are detected; two concurrent
\*    transactions writing the same key cannot both commit.
NoLostUpdates ==
    \A t1, t2 \in Tx :
        (t1 /= t2
         /\ txState[t1] = "committed"
         /\ txState[t2] = "committed"
         /\ writeSet[t1] \cap writeSet[t2] /= {})
        => \* One must have seen the other's write (commit ordering)
           \/ commitTS[t1] <= readTS[t2]
           \/ commitTS[t2] <= readTS[t1]

\* 3. Snapshot Consistency: each transaction reads from a consistent
\*    point-in-time snapshot. All visible versions for a given transaction
\*    have writeTs <= readTS[t] and are committed.
SnapshotConsistency ==
    \A t \in Tx :
        txState[t] \in {"active", "committed"} =>
            \A k \in Key :
                LET vis == VisibleVersion(k, t) IN
                vis /= Nil =>
                    /\ vis.committed = TRUE
                    /\ vis.writeTs <= readTS[t]

\* 4. Read-Your-Writes: a transaction sees its own uncommitted writes.
\*    If t wrote key k, then OwnVersion(k, t) is non-Nil while t is active.
ReadYourWrites ==
    \A t \in Tx :
        txState[t] = "active" =>
            \A k \in writeSet[t] :
                OwnVersion(k, t) /= Nil

\* Combined safety invariant for model checking
SafetyInvariant ==
    /\ NoDirtyReads
    /\ NoLostUpdates
    /\ SnapshotConsistency
    /\ ReadYourWrites

\* Type invariant
TypeOK ==
    /\ txState \in [Tx -> {"idle", "active", "committed", "aborted"}]
    /\ readTS \in [Tx -> 0..MaxTS]
    /\ commitTS \in [Tx -> 0..MaxTS]
    /\ \A t \in Tx : writeSet[t] \subseteq Key
    /\ \A t \in Tx : readSet[t] \subseteq Key
    /\ tsOracle \in 0..MaxTS

=============================================================================
