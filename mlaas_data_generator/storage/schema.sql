PRAGMA foreign_keys = ON;

-- =========================================================
-- Runs
-- =========================================================
CREATE TABLE IF NOT EXISTS runs (
  run_id              TEXT PRIMARY KEY,
  dataset             TEXT NOT NULL,
  task_type           TEXT NOT NULL CHECK (task_type IN ('classification','regression','clustering')),
  model_type          TEXT NOT NULL,

  -- split / distribution
  split_strategy      TEXT NOT NULL,
  distribution_param  TEXT,

  -- run knobs
  num_clients         INTEGER NOT NULL,
  num_rounds          INTEGER NOT NULL,
  seed                INTEGER,
  save_weights        INTEGER NOT NULL DEFAULT 0,

  -- training knobs
  learning_rate       REAL,
  batch_size          INTEGER,
  local_epochs        INTEGER,
  hidden_layers       TEXT,
  activation          TEXT,
  dropout             REAL,
  weight_decay        REAL,
  optimizer           TEXT,

  -- model/meta
  params_count        INTEGER,
  metric_name         TEXT,
  distribution_bins   INTEGER,

  -- clustering passthroughs
  clustering_k        INTEGER,
  clustering_init     TEXT,
  clustering_n_init   INTEGER,
  clustering_max_iter INTEGER,
  clustering_tol      REAL,

  -- misc
  dataset_args_json   TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_dataset       ON runs(dataset);
CREATE INDEX IF NOT EXISTS idx_runs_task          ON runs(task_type);
CREATE INDEX IF NOT EXISTS idx_runs_model         ON runs(model_type);
CREATE INDEX IF NOT EXISTS idx_runs_split         ON runs(split_strategy);

-- =========================================================
-- Rounds (one row per round)
-- =========================================================
CREATE TABLE IF NOT EXISTS rounds (
  run_id              TEXT NOT NULL,
  round               INTEGER NOT NULL,
  global_loss         REAL,
  global_metric       REAL,
  global_metric_name  TEXT NOT NULL,   -- 'accuracy' | 'rmse' | 'silhouette'
  global_aux_metric   REAL,            -- e.g., f1 or inertia
  global_score        REAL,
  frontier_json       TEXT,
  PRIMARY KEY (run_id, round),
  FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_rounds_metric      ON rounds(run_id, global_metric);
CREATE INDEX IF NOT EXISTS idx_rounds_score       ON rounds(run_id, global_score);

-- =========================================================
-- Client rounds (one row per client per round)
-- =========================================================
CREATE TABLE IF NOT EXISTS client_rounds (
  run_id                       TEXT NOT NULL,
  round                        INTEGER NOT NULL,
  client_id                    TEXT NOT NULL,

  participated                 INTEGER NOT NULL,  -- 0/1
  round_fail_reason            TEXT,
  rounds_participated_so_far   INTEGER,

  data_distribution_json       TEXT,
  samples_count                INTEGER,
  computation_time_s           REAL,

  comm_bytes_up                INTEGER,
  comm_bytes_down              INTEGER,

  -- QoS / metrics (task-dependent; nullable)
  loss                         REAL,
  accuracy                     REAL,
  f1                           REAL,
  rmse                         REAL,
  rmse_original_units          REAL,
  silhouette                   REAL,
  inertia                      REAL,

  metric_score                 REAL,
  extra_metric                 REAL,

  -- optional clustering extras
  ari                          REAL,
  nmi                          REAL,
  clustering_k                 INTEGER,
  clustering_agg               TEXT,
  
  availability_flag            INTEGER,  -- 0/1
  PRIMARY KEY (run_id, round, client_id),
  FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
  -- If you later want to enforce (run_id, round) existence, you can add:
  -- , FOREIGN KEY (run_id, round) REFERENCES rounds(run_id, round) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX IF NOT EXISTS idx_client_rounds_run_round  ON client_rounds(run_id, round);
CREATE INDEX IF NOT EXISTS idx_client_rounds_client     ON client_rounds(run_id, client_id);
CREATE INDEX IF NOT EXISTS idx_client_rounds_metric     ON client_rounds(run_id, round, metric_score);
CREATE INDEX IF NOT EXISTS idx_client_rounds_accuracy   ON client_rounds(run_id, round, accuracy);
CREATE INDEX IF NOT EXISTS idx_client_rounds_rmse       ON client_rounds(run_id, round, rmse);
CREATE INDEX IF NOT EXISTS idx_client_rounds_silhouette ON client_rounds(run_id, round, silhouette);
