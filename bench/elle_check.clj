;; External linearizability / strict-serializability history checker using official `elle/elle` (0.2.8).
;; Usage:
;;   clojure -Sdeps '{:mvn/repos {"clojars" {:url "https://repo.clojars.org/"}} :deps {elle/elle {:mvn/version "0.2.8"} spootnik/unilog {:mvn/version "0.7.31"}}}' \
;;     -M bench/elle_check.clj bench/results/raft_tcp_elle_history.edn rw-register

(ns elle-check
  (:require [clojure.edn :as edn]
            [clojure.pprint :as pprint]
            [jepsen.history :as h]
            [elle.rw-register :as rw]
            [elle.list-append :as la]))

(let [path (first *command-line-args*)
      _    (when-not path
             (binding [*out* *err*]
               (println "Usage: elle_check.clj <history.edn> [rw-register|list-append]"))
             (System/exit 2))
      mode (or (second *command-line-args*) "rw-register")
      raw  (slurp path)
      hist (h/history (edn/read-string raw))
      res  (case mode
             "rw-register" (rw/check {:consistency-models [:strict-serializable]} hist)
             "list-append" (la/check {:consistency-models [:strict-serializable]} hist)
             (throw (ex-info (str "Unknown mode: " mode) {:mode mode})))]
  (println "=== Official jepsen/elle 0.2.8 Verification ===")
  (println "History file :" path)
  (println "Model        :" mode "(:strict-serializable)")
  (println "Events count :" (count hist))
  (pprint/pprint res)
  (System/exit (if (:valid? res) 0 1)))
