-- 1) Runs: one row per run (your “knobs” at start time)
CREATE TABLE runs (
  run_id              TEXT PRIMARY KEY,           -- UUID string from your code
  created_at          TIMESTAMP NOT NULL,         -- set when run starts
  dataset             TEXT NOT NULL,
  task_type           TEXT NOT NULL CHECK (task_type IN ('classification','regression','clustering')),
  model_type          TEXT NOT NULL,              -- CNN/MLP/... (metadata for clustering)
  split_strategy      TEXT NOT NULL,              -- iid, dirichlet, shard, ...
  distribution_param  TEXT,                       -- store as text to allow float/int
  num_clients         INTEGER NOT NULL,
  num_rounds          INTEGER NOT NULL,
  seed                INTEGER,
  save_weights        BOOLEAN NOT NULL DEFAULT 0,

  -- optional run-level knobs (nullable when irrelevant)
  learning_rate       REAL,
  batch_size          INTEGER,
  local_epochs        INTEGER,
  hidden_layers       TEXT,                       -- "128,64" (simple, parse later)
  activation          TEXT,
  dropout             REAL,
  weight_decay        REAL,
  optimizer           TEXT,

  -- clustering knobs (nullable for non-clustering)
  clustering_k        INTEGER,
  clustering_init     TEXT,
  clustering_n_init   INTEGER,
  clustering_max_iter INTEGER,
  clustering_tol      REAL,

  -- dataset args (keep simple & flexible)
  dataset_args_json   TEXT                        -- JSON string of your dataset_args
);

CREATE INDEX idx_runs_dataset ON runs(dataset);
CREATE INDEX idx_runs_task ON runs(task_type);


-- 2) Rounds: one row per global round (aggregates + global metric)
CREATE TABLE rounds (
  run_id              TEXT NOT NULL,
  round               INTEGER NOT NULL,           -- 1..num_rounds
  -- global metrics (nullable depending on task)
  global_loss         REAL,
  global_metric       REAL,                       -- accuracy / rmse / silhouette (primary)
  global_metric_name  TEXT NOT NULL,              -- 'accuracy' | 'rmse' | 'silhouette'
  global_aux_metric   REAL,                       -- f1 (cls) | inertia (clu) | null
  global_score        REAL,                       -- your 1/(1+rmse) mapping etc.
  -- optional summaries
  frontier_json       TEXT,                       -- optional skyline summary per round

  PRIMARY KEY (run_id, round),
  FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX idx_rounds_metric ON rounds(run_id, global_metric);


-- 3) Clients: one row per client per round (your existing per-client record)
CREATE TABLE client_rounds (
  run_id                       TEXT NOT NULL,
  round                        INTEGER NOT NULL,
  client_id                    TEXT NOT NULL,

  -- participation / reliability signals
  participated                 BOOLEAN NOT NULL,
  round_fail_reason            TEXT,              -- '', 'error', 'dropped_out', ...
  rounds_participated_so_far   INTEGER,           -- monotone per client

  -- data characteristics
  data_distribution_json       TEXT,              -- label hist or regression bins
  samples_count                INTEGER,

  -- QoS/time/comms
  computation_time_s           REAL,              -- your Computation_Time
  comm_bytes_up                INTEGER,
  comm_bytes_down              INTEGER,

  -- loss + metrics (nullable depending on task)
  loss                         REAL,
  accuracy                     REAL,              -- classification primary
  f1                           REAL,              -- classification aux
  rmse                         REAL,              -- regression primary
  rmse_original_units          REAL,              -- if you add back-transform
  silhouette                   REAL,              -- clustering primary
  inertia                      REAL,              -- clustering aux
  metric_score                 REAL,              -- your unified “score” field
  extra_metric                 REAL,              -- keep for backward compat

  -- clustering extras (nullable)
  ari                          REAL,
  nmi                          REAL,
  clustering_k                 INTEGER,
  clustering_agg               TEXT,              -- 'local_only' etc.

  -- availability/cost/throughput hooks (optional, future-proof)
  availability_flag            INTEGER,           -- 1 if participated else 0
  throughput_eps               REAL,
  inference_latency_ms_mean    REAL,
  inference_latency_ms_p95     REAL,
  compute_cost_usd             REAL,
  total_cost_usd               REAL,

  -- reliability (over rounds), denormalized convenience
  reliability_score            REAL,

  PRIMARY KEY (run_id, round, client_id),
  FOREIGN KEY (run_id, round) REFERENCES rounds(run_id, round) ON DELETE CASCADE
);

CREATE INDEX idx_client_rounds_run_round ON client_rounds(run_id, round);
CREATE INDEX idx_client_rounds_client ON client_rounds(run_id, client_id);
CREATE INDEX idx_client_rounds_metric ON client_rounds(run_id, round, metric_score);
