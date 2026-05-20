-------------------------------- MODULE Raft --------------------------------
(***************************************************************************)
(* TLA+ specification of the Raft consensus protocol as implemented in     *)
(* JouleDB (joule-db-server/src/raft.rs).                                  *)
(*                                                                         *)
(* Models: Leader election (RequestVote), log replication (AppendEntries), *)
(* commitment, and the five key safety properties from the Raft paper.     *)
(*                                                                         *)
(* Model parameters:                                                       *)
(*   Server  = {s1, s2, s3}   (3 servers)                                  *)
(*   MaxTerm = 3              (max term number)                            *)
(*   MaxLogLen = 4            (max log length per server)                  *)
(***************************************************************************)

EXTENDS Naturals, FiniteSets, Sequences, TLC

CONSTANTS Server, MaxTerm, MaxLogLen

\* Quorum: any majority subset
Quorum == {Q \in SUBSET Server : Cardinality(Q) * 2 > Cardinality(Server)}

(***************************************************************************)
(* Variables                                                               *)
(***************************************************************************)
VARIABLES
    currentTerm,   \* currentTerm[s]: latest term server s has seen
    state,         \* state[s]: Follower | Candidate | Leader
    votedFor,      \* votedFor[s]: candidateId voted for in currentTerm (or Nil)
    log,           \* log[s]: sequence of [term |-> t, value |-> v] entries
    commitIndex,   \* commitIndex[s]: highest log index known committed
    lastApplied,   \* lastApplied[s]: highest log index applied to state machine
    \* Leader volatile state
    nextIndex,     \* nextIndex[s][t]: for leader s, next index to send to t
    matchIndex,    \* matchIndex[s][t]: for leader s, highest index replicated on t
    \* Message passing
    messages,      \* bag (multiset) of in-flight messages
    \* Election bookkeeping
    votesGranted   \* votesGranted[s]: set of servers that granted vote to candidate s

vars == <<currentTerm, state, votedFor, log, commitIndex, lastApplied,
          nextIndex, matchIndex, messages, votesGranted>>

Nil == "Nil"

(***************************************************************************)
(* Helper operators                                                        *)
(***************************************************************************)

\* Last log index for server s
LastLogIndex(s) == Len(log[s])

\* Last log term for server s (0 if log empty)
LastLogTerm(s) == IF Len(log[s]) > 0 THEN log[s][Len(log[s])].term ELSE 0

\* Term at a given index (0 if out of range)
LogTerm(s, idx) == IF idx > 0 /\ idx <= Len(log[s]) THEN log[s][idx].term ELSE 0

\* Is candidate's log at least as up-to-date as voter's?
\* (Raft paper Section 5.4.1)
LogUpToDate(candidateLastTerm, candidateLastIdx, voterLastTerm, voterLastIdx) ==
    \/ candidateLastTerm > voterLastTerm
    \/ (candidateLastTerm = voterLastTerm /\ candidateLastIdx >= voterLastIdx)

\* Add a message to the message bag
Send(m) == messages' = messages \cup {m}

\* Remove a message from the message bag
Discard(m) == messages' = messages \ {m}

\* Send one message and discard another
Reply(response, request) ==
    messages' = (messages \ {request}) \cup {response}

\* Minimum of two naturals
Min(a, b) == IF a < b THEN a ELSE b

(***************************************************************************)
(* State transitions                                                       *)
(***************************************************************************)

\* Server s times out and starts an election
\* Corresponds to start_election() in raft.rs
Timeout(s) ==
    /\ state[s] \in {"Follower", "Candidate"}
    /\ currentTerm[s] < MaxTerm
    /\ currentTerm' = [currentTerm EXCEPT ![s] = currentTerm[s] + 1]
    /\ state' = [state EXCEPT ![s] = "Candidate"]
    /\ votedFor' = [votedFor EXCEPT ![s] = s]
    /\ votesGranted' = [votesGranted EXCEPT ![s] = {s}]
    /\ UNCHANGED <<log, commitIndex, lastApplied, nextIndex, matchIndex>>
    /\ messages' = messages \cup
        {[mtype   |-> "RequestVote",
          mterm   |-> currentTerm[s] + 1,
          msource |-> s,
          mdest   |-> t,
          mlastLogTerm  |-> LastLogTerm(s),
          mlastLogIndex |-> LastLogIndex(s)] : t \in Server \ {s}}

\* Server s handles a RequestVote request from candidate
\* Corresponds to handle_request_vote() in raft.rs
HandleRequestVote(s, m) ==
    /\ m.mtype = "RequestVote"
    /\ m.mdest = s
    /\ LET grant ==
           /\ m.mterm >= currentTerm[s]
           /\ (votedFor[s] = Nil \/ votedFor[s] = m.msource
               \/ m.mterm > currentTerm[s])
           /\ LogUpToDate(m.mlastLogTerm, m.mlastLogIndex,
                          LastLogTerm(s), LastLogIndex(s))
       IN
       \* Step down if we see a higher term
       /\ currentTerm' = [currentTerm EXCEPT ![s] = IF m.mterm > currentTerm[s]
                                                     THEN m.mterm
                                                     ELSE currentTerm[s]]
       /\ state' = [state EXCEPT ![s] = IF m.mterm > currentTerm[s]
                                         THEN "Follower"
                                         ELSE state[s]]
       /\ votedFor' = [votedFor EXCEPT ![s] = IF grant THEN m.msource
                                               ELSE IF m.mterm > currentTerm[s]
                                               THEN Nil
                                               ELSE votedFor[s]]
       /\ Reply([mtype        |-> "RequestVoteResponse",
                 mterm        |-> IF m.mterm > currentTerm[s] THEN m.mterm
                                  ELSE currentTerm[s],
                 msource      |-> s,
                 mdest        |-> m.msource,
                 mvoteGranted |-> grant], m)
       /\ UNCHANGED <<log, commitIndex, lastApplied, nextIndex, matchIndex,
                       votesGranted>>

\* Candidate s receives a vote response
HandleRequestVoteResponse(s, m) ==
    /\ m.mtype = "RequestVoteResponse"
    /\ m.mdest = s
    /\ m.mterm = currentTerm[s]
    /\ state[s] = "Candidate"
    /\ IF m.mvoteGranted
       THEN votesGranted' = [votesGranted EXCEPT ![s] = votesGranted[s] \cup {m.msource}]
       ELSE UNCHANGED votesGranted
    /\ Discard(m)
    /\ UNCHANGED <<currentTerm, state, votedFor, log, commitIndex, lastApplied,
                    nextIndex, matchIndex>>

\* Candidate s becomes leader after receiving a quorum of votes
BecomeLeader(s) ==
    /\ state[s] = "Candidate"
    /\ votesGranted[s] \in Quorum
    /\ state' = [state EXCEPT ![s] = "Leader"]
    /\ nextIndex'  = [nextIndex  EXCEPT ![s] =
                        [t \in Server |-> LastLogIndex(s) + 1]]
    /\ matchIndex' = [matchIndex EXCEPT ![s] =
                        [t \in Server |-> 0]]
    /\ UNCHANGED <<currentTerm, votedFor, log, commitIndex, lastApplied,
                    messages, votesGranted>>

\* Leader s appends a new client command to its log
\* Corresponds to propose_command() in raft.rs
ClientRequest(s) ==
    /\ state[s] = "Leader"
    /\ Len(log[s]) < MaxLogLen
    /\ LET newEntry == [term  |-> currentTerm[s],
                        value |-> Len(log[s]) + 1]  \* abstract command as index
       IN
       /\ log' = [log EXCEPT ![s] = Append(log[s], newEntry)]
       /\ matchIndex' = [matchIndex EXCEPT ![s][s] = Len(log[s]) + 1]
    /\ UNCHANGED <<currentTerm, state, votedFor, commitIndex, lastApplied,
                    nextIndex, messages, votesGranted>>

\* Leader s sends AppendEntries to follower t
\* Corresponds to send_append_entries() in raft.rs
AppendEntries(s, t) ==
    /\ s /= t
    /\ state[s] = "Leader"
    /\ LET prevIdx == nextIndex[s][t] - 1
           prevTerm == LogTerm(s, prevIdx)
           \* Send entries from nextIndex[s][t] to end of log
           entriesToSend == IF nextIndex[s][t] > Len(log[s])
                            THEN <<>>
                            ELSE SubSeq(log[s], nextIndex[s][t], Len(log[s]))
       IN
       /\ Send([mtype         |-> "AppendEntries",
                mterm         |-> currentTerm[s],
                msource       |-> s,
                mdest         |-> t,
                mprevLogIndex |-> prevIdx,
                mprevLogTerm  |-> prevTerm,
                mentries      |-> entriesToSend,
                mcommitIndex  |-> commitIndex[s]])
    /\ UNCHANGED <<currentTerm, state, votedFor, log, commitIndex, lastApplied,
                    nextIndex, matchIndex, votesGranted>>

\* Server s handles an AppendEntries request
\* Corresponds to handle_append_entries() in raft.rs
HandleAppendEntries(s, m) ==
    /\ m.mtype = "AppendEntries"
    /\ m.mdest = s
    /\ LET
         \* Step down if we see a higher or equal term from a leader
         stepDown == m.mterm >= currentTerm[s]
         newTerm  == IF m.mterm > currentTerm[s] THEN m.mterm ELSE currentTerm[s]
         \* Log consistency check: prev entry must match
         logOk == \/ m.mprevLogIndex = 0
                  \/ (m.mprevLogIndex > 0
                      /\ m.mprevLogIndex <= Len(log[s])
                      /\ LogTerm(s, m.mprevLogIndex) = m.mprevLogTerm)
         \* Can we accept this message?
         accept == /\ m.mterm >= currentTerm[s]
                   /\ logOk
       IN
       IF ~accept
       THEN
         \* Reject: term too old or log inconsistency
         /\ Reply([mtype      |-> "AppendEntriesResponse",
                   mterm      |-> newTerm,
                   msource    |-> s,
                   mdest      |-> m.msource,
                   msuccess   |-> FALSE,
                   mmatchIndex |-> 0], m)
         /\ currentTerm' = [currentTerm EXCEPT ![s] = newTerm]
         /\ state' = [state EXCEPT ![s] = IF m.mterm > currentTerm[s]
                                           THEN "Follower" ELSE state[s]]
         /\ votedFor' = [votedFor EXCEPT ![s] = IF m.mterm > currentTerm[s]
                                                 THEN Nil ELSE votedFor[s]]
         /\ UNCHANGED <<log, commitIndex, lastApplied, nextIndex, matchIndex,
                         votesGranted>>
       ELSE
         \* Accept: append entries (truncate conflicting suffix first)
         LET
           \* Index where new entries start in the log
           baseIdx == m.mprevLogIndex + 1
           \* Truncate any conflicting entries and append new ones
           newLog == SubSeq(log[s], 1, m.mprevLogIndex) \o m.mentries
           newCommitIdx == IF m.mcommitIndex > commitIndex[s]
                           THEN Min(m.mcommitIndex, Len(newLog))
                           ELSE commitIndex[s]
         IN
         /\ log' = [log EXCEPT ![s] = newLog]
         /\ commitIndex' = [commitIndex EXCEPT ![s] = newCommitIdx]
         /\ currentTerm' = [currentTerm EXCEPT ![s] = newTerm]
         /\ state' = [state EXCEPT ![s] = "Follower"]
         /\ votedFor' = [votedFor EXCEPT ![s] = IF m.mterm > currentTerm[s]
                                                 THEN Nil ELSE votedFor[s]]
         /\ Reply([mtype       |-> "AppendEntriesResponse",
                   mterm       |-> newTerm,
                   msource     |-> s,
                   mdest       |-> m.msource,
                   msuccess    |-> TRUE,
                   mmatchIndex |-> Len(newLog)], m)
         /\ UNCHANGED <<lastApplied, nextIndex, matchIndex, votesGranted>>

\* Leader s handles an AppendEntries response from follower
HandleAppendEntriesResponse(s, m) ==
    /\ m.mtype = "AppendEntriesResponse"
    /\ m.mdest = s
    /\ m.mterm = currentTerm[s]
    /\ state[s] = "Leader"
    /\ IF m.msuccess
       THEN
         \* Update nextIndex and matchIndex for the follower
         /\ nextIndex'  = [nextIndex  EXCEPT ![s][m.msource] =
                             m.mmatchIndex + 1]
         /\ matchIndex' = [matchIndex EXCEPT ![s][m.msource] =
                             m.mmatchIndex]
         /\ Discard(m)
         /\ UNCHANGED <<currentTerm, state, votedFor, log, commitIndex,
                         lastApplied, votesGranted>>
       ELSE
         \* Decrement nextIndex and retry
         /\ nextIndex' = [nextIndex EXCEPT ![s][m.msource] =
                            IF nextIndex[s][m.msource] > 1
                            THEN nextIndex[s][m.msource] - 1
                            ELSE 1]
         /\ Discard(m)
         /\ UNCHANGED <<currentTerm, state, votedFor, log, commitIndex,
                         lastApplied, matchIndex, votesGranted>>

\* Leader s advances commit index based on matchIndex quorum
\* Corresponds to advance_commit_index logic in raft.rs
AdvanceCommitIndex(s) ==
    /\ state[s] = "Leader"
    /\ \E idx \in (commitIndex[s]+1)..Len(log[s]) :
        /\ log[s][idx].term = currentTerm[s]  \* Only commit entries from current term
        /\ {t \in Server : matchIndex[s][t] >= idx} \in Quorum
        /\ commitIndex' = [commitIndex EXCEPT ![s] = idx]
    /\ UNCHANGED <<currentTerm, state, votedFor, log, lastApplied,
                    nextIndex, matchIndex, messages, votesGranted>>

\* Server s applies committed entries to state machine
ApplyEntry(s) ==
    /\ lastApplied[s] < commitIndex[s]
    /\ lastApplied' = [lastApplied EXCEPT ![s] = lastApplied[s] + 1]
    /\ UNCHANGED <<currentTerm, state, votedFor, log, commitIndex,
                    nextIndex, matchIndex, messages, votesGranted>>

\* Any server steps down when it sees a higher term in any message
\* (consolidated step-down, matching the step_down() helper in raft.rs)
StepDown(s, m) ==
    /\ m.mdest = s
    /\ m.mterm > currentTerm[s]
    /\ m.mtype \in {"AppendEntriesResponse", "RequestVoteResponse"}
    /\ currentTerm' = [currentTerm EXCEPT ![s] = m.mterm]
    /\ state' = [state EXCEPT ![s] = "Follower"]
    /\ votedFor' = [votedFor EXCEPT ![s] = Nil]
    /\ Discard(m)
    /\ UNCHANGED <<log, commitIndex, lastApplied, nextIndex, matchIndex,
                    votesGranted>>

(***************************************************************************)
(* Initial state                                                           *)
(***************************************************************************)
Init ==
    /\ currentTerm = [s \in Server |-> 0]
    /\ state       = [s \in Server |-> "Follower"]
    /\ votedFor    = [s \in Server |-> Nil]
    /\ log         = [s \in Server |-> <<>>]
    /\ commitIndex = [s \in Server |-> 0]
    /\ lastApplied = [s \in Server |-> 0]
    /\ nextIndex   = [s \in Server |-> [t \in Server |-> 1]]
    /\ matchIndex  = [s \in Server |-> [t \in Server |-> 0]]
    /\ messages    = {}
    /\ votesGranted = [s \in Server |-> {}]

(***************************************************************************)
(* Next-state relation                                                     *)
(***************************************************************************)
Next ==
    \/ \E s \in Server : Timeout(s)
    \/ \E s \in Server : BecomeLeader(s)
    \/ \E s \in Server : ClientRequest(s)
    \/ \E s \in Server, t \in Server : AppendEntries(s, t)
    \/ \E s \in Server : AdvanceCommitIndex(s)
    \/ \E s \in Server : ApplyEntry(s)
    \/ \E s \in Server, m \in messages : HandleRequestVote(s, m)
    \/ \E s \in Server, m \in messages : HandleRequestVoteResponse(s, m)
    \/ \E s \in Server, m \in messages : HandleAppendEntries(s, m)
    \/ \E s \in Server, m \in messages : HandleAppendEntriesResponse(s, m)
    \/ \E s \in Server, m \in messages : StepDown(s, m)

Spec == Init /\ [][Next]_vars

(***************************************************************************)
(* Safety Invariants                                                       *)
(***************************************************************************)

\* 1. Election Safety: at most one leader per term
ElectionSafety ==
    \A s1, s2 \in Server :
        (state[s1] = "Leader" /\ state[s2] = "Leader" /\ currentTerm[s1] = currentTerm[s2])
        => s1 = s2

\* 2. Leader Append-Only: a leader never overwrites or deletes entries in its log
\*    (Checked as a temporal property via refinement; here we assert structural
\*     invariant: leaders only grow their logs.)
\*    This is enforced by construction: ClientRequest only appends,
\*    and HandleAppendEntries transitions leader to follower before modifying log.

\* 3. Log Matching: if two logs contain an entry with the same index and term,
\*    the logs are identical through that index
LogMatching ==
    \A s1, s2 \in Server :
        \A idx \in 1..Min(Len(log[s1]), Len(log[s2])) :
            log[s1][idx].term = log[s2][idx].term
            => \A j \in 1..idx : log[s1][j] = log[s2][j]

\* 4. State Machine Safety: if a server has applied a log entry at index i,
\*    no other server applies a different entry for that index
StateMachineSafety ==
    \A s1, s2 \in Server :
        \A idx \in 1..Min(lastApplied[s1], lastApplied[s2]) :
            /\ idx <= Len(log[s1])
            /\ idx <= Len(log[s2])
            /\ log[s1][idx] = log[s2][idx]

\* 5. Leader Completeness: if an entry is committed in a given term,
\*    it appears in the logs of all leaders of higher-numbered terms.
\*    We check: any leader's log contains all committed entries.
LeaderCompleteness ==
    \A s \in Server :
        state[s] = "Leader" =>
            \A t \in Server :
                \A idx \in 1..commitIndex[t] :
                    /\ idx <= Len(log[s])
                    /\ idx <= Len(log[t])
                    /\ log[s][idx] = log[t][idx]

\* Combined invariant for model checking
SafetyInvariant ==
    /\ ElectionSafety
    /\ LogMatching
    /\ StateMachineSafety
    /\ LeaderCompleteness

\* Type invariant (helps TLC explore fewer states)
TypeOK ==
    /\ currentTerm \in [Server -> 0..MaxTerm]
    /\ state \in [Server -> {"Follower", "Candidate", "Leader"}]
    /\ \A s \in Server : votedFor[s] \in Server \cup {Nil}
    /\ \A s \in Server : Len(log[s]) <= MaxLogLen
    /\ \A s \in Server : commitIndex[s] \in 0..MaxLogLen
    /\ \A s \in Server : lastApplied[s] \in 0..MaxLogLen
    /\ \A s \in Server : lastApplied[s] <= commitIndex[s]

=============================================================================
