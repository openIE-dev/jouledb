//! # Joule Web
//!
//! Energy-efficient web application framework in pure Rust.
//! Compiles to WASM (398 KB) and replaces the JavaScript/TypeScript library
//! ecosystem with a single, type-safe, energy-tracked crate.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │  joule-web                                          │
//! │                                                     │
//! │  Core Rendering          Application Logic          │
//! │  ├── vdom                ├── router                 │
//! │  ├── reactive            ├── state                  │
//! │  ├── component           ├── forms                  │
//! │  └── html                └── fetch                  │
//! │                                                     │
//! │  UI & Presentation       Platform                   │
//! │  ├── css                 ├── storage                │
//! │  ├── animation           └── energy                 │
//! │  ├── a11y                                           │
//! │  └── i18n                                           │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! All modules are pure Rust — no `wasm-bindgen` or `web-sys` in the core.
//! Browser bindings are injected at the application boundary, keeping
//! the entire framework testable on native targets.

// Numerical / rendering code uses `let mut x = 0.0;` + loop-overwrite
// idioms (Bresenham-style rasterizers, bidi walkers, SVG path cursors,
// orbital-mechanics state). Bidi/BIDI tables also have overlapping
// Unicode codepoint ranges where later arms are unreachable but kept
// for documentation parity with the spec.
#![allow(unused_assignments, non_snake_case, unreachable_patterns)]

// ── Core Rendering ──
pub mod component;
pub mod html;
pub mod reactive;
pub mod vdom;

// ── Application Logic ──
pub mod fetch;
pub mod forms;
pub mod router;
pub mod state;

// ── UI & Presentation ──
pub mod a11y;
pub mod animation;
pub mod css;
pub mod i18n;

// ── Interaction ──
pub mod drag;
pub mod gesture;
pub mod hotkey;

// ── Data & Display ──
pub mod chart;
pub mod chart_export;
pub mod clustermap;
pub mod facet;
pub mod highdim;
pub mod interact;
pub mod markdown;
pub mod stat_chart;
pub mod stat_overlay;
pub mod surface3d;
pub mod table;
pub mod virtualize;

// ── Utilities ──
pub mod color;
pub mod date;
pub mod schema;

// ── Async & Realtime ──
pub mod query;
pub mod websocket;

// ── Layout & Rendering ──
pub mod head;
pub mod portal;
pub mod theme;
pub mod toast;

// ── Graphics ──
pub mod canvas2d;
pub mod webgl;

// ── Communication ──
pub mod clipboard;
pub mod graphql;
pub mod sse;
pub mod worker;

// ── Security ──
pub mod auth;
pub mod crypto;
pub mod sanitize;

// ── Data Formats ──
pub mod csv;
pub mod highlight;
pub mod search;

// ── Navigation & UX ──
pub mod pwa;
pub mod scroll;
pub mod transition;
pub mod upload;

// ── Browser APIs ──
pub mod abort;
pub mod broadcast;
pub mod gamepad;
pub mod geo;
pub mod indexeddb;
pub mod intersection;
pub mod lock;
pub mod notification;
pub mod resize;
pub mod speech;
pub mod streams;
pub mod webrtc;

// ── Data Processing ──
pub mod diff;
pub mod image;
pub mod json_patch;
pub mod pdf;
pub mod qrcode;
pub mod xml;
pub mod zip;

// ── UI Patterns ──
pub mod accordion;
pub mod breadcrumb;
pub mod carousel;
pub mod command_palette;
pub mod floating;
pub mod mask;
pub mod skeleton;
pub mod tabs;
pub mod tree;

// ── Infrastructure ──
pub mod analytics;
pub mod error_track;
pub mod feature_flag;
pub mod fsm;
pub mod immutable;
pub mod observable;
pub mod pipe;
pub mod rate_limit;
pub mod template;
pub mod test_utils;

// ── 3D Engine ──
pub mod camera3d;
pub mod lighting;
pub mod material;
pub mod mesh;
pub mod particle;
pub mod physics3d;
pub mod scene3d;
pub mod terrain;

// ── Audio ──
pub mod audio_analysis;
pub mod audio_buffer;
pub mod audio_engine;
pub mod audio_filter;
pub mod audio_fx;
pub mod audio_graph;
pub mod delay_line;
pub mod envelope;
pub mod equalizer;
pub mod midi;
pub mod oscillator;
pub mod spatial_audio;
pub mod synthesizer;
pub mod waveform;

// ── Video & Media ──
pub mod hls_stream;
pub mod media_recorder;
pub mod media_session;
pub mod playlist;
pub mod screen_capture;
pub mod subtitle;
pub mod video_player;
pub mod webcodecs;

// ── ML & AI ──
pub mod embedding;
pub mod face_detect;
pub mod model_loader;
pub mod nlp_token;
pub mod pose_detect;
pub mod segmentation;
pub mod tensor;
pub mod vision_pipeline;

// ── Maps & Advanced Viz ──
pub mod axis_render;
pub mod force_layout;
pub mod geo_projection;
pub mod map_engine;
pub mod marker_cluster;
pub mod scale;
pub mod tile_renderer;
pub mod voronoi;

// ── Rich Text & Editors ──
pub mod autocomplete;
pub mod code_format;
pub mod diff_editor;
pub mod doc_model;
pub mod rich_editor;
pub mod spell_check;
pub mod syntax_editor;
pub mod text_selection;

// ── Diagramming ──
pub mod diagram;
pub mod flowchart;
pub mod gantt;
pub mod graph_layout;
pub mod mindmap;
pub mod network_viz;
pub mod org_chart;
pub mod sankey;

// ── File Formats ──
pub mod docx_gen;
pub mod font_loader;
pub mod opentype;
pub mod xlsx;

// ── Advanced Auth & P2P ──
pub mod jwt_codec;
pub mod oauth2_pkce;
pub mod p2p_mesh;
pub mod webauthn;

// ── Currency ──
pub mod currency;

// ── Typography & Text ──
pub mod bidi;
pub mod font_metrics;
pub mod hyphenation;
pub mod ligature;
pub mod line_break;
pub mod text_shaper;
pub mod text_wrap;
pub mod unicode_normalize;

// ── Advanced Charts ──
pub mod candlestick;
pub mod funnel;
pub mod gauge;
pub mod graph_chart;
pub mod parallel_coords;
pub mod polar_chart;
pub mod radar_chart;
pub mod scatter3d;

// ── Testing Framework ──
pub mod assertion;
pub mod benchmark;
pub mod coverage;
pub mod coverage_tracker;
pub mod dom_test;
pub mod fault_inject;
pub mod fixture;
pub mod mock_builder;
pub mod mock_http;
pub mod perf_bench;
pub mod property_test;
pub mod snapshot_test;
pub mod test_data_gen;
pub mod test_runner;
pub mod visual_regression;

// ── Build Tools ──
pub mod asset_pipeline;
pub mod bundler;
pub mod css_modules;
pub mod hot_reload;
pub mod minifier;
pub mod source_map;
pub mod transpiler;
pub mod tree_shake;

// ── CSS Layout Engine ──
pub mod box_model;
pub mod cascade_resolver;
pub mod css_grid;
pub mod css_transition_engine;
pub mod css_var;
pub mod flexbox;
pub mod selector_engine;
pub mod specificity;

// ── Database & Storage ──
pub mod cache_strategy;
pub mod connection_pool;
pub mod data_sync;
pub mod migration;
pub mod offline_store;
pub mod orm;
pub mod query_builder;
pub mod schema_validate;

// ── Database & Query Patterns ──
pub mod aggregation;
pub mod btree_index;
pub mod cursor;
pub mod hash_index;
pub mod join_algorithms;
pub mod query_planner;
pub mod transaction_log;

// ── Cache & Storage Patterns ──
pub mod buffer_pool;
pub mod content_cache;
pub mod kv_store;
pub mod memory_pool;
pub mod object_pool;
pub mod session_store;
pub mod write_ahead_log;

// ── DevTools & Monitoring ──
pub mod call_graph;
pub mod console_logger;
pub mod debug_protocol;
pub mod debugger;
pub mod error_tracker;
pub mod flame_graph;
pub mod heap_snapshot;
pub mod inspector;
pub mod invariant_checker;
pub mod lighthouse;
pub mod memory_profiler;
pub mod network_monitor;
pub mod profiler;
pub mod profiler_report;

// ── Animation & Motion ──
pub mod keyframe_anim;
pub mod lottie;
pub mod morph;
pub mod motion_path;
pub mod parallax;
pub mod scroll_anim;
pub mod spring_anim;
pub mod tween;

// ── Advanced State & Reactivity ──
pub mod atom;
pub mod computed;
pub mod data_binding;
pub mod devtools_state;
pub mod effect;
pub mod event_emitter;
pub mod finite_state;
pub mod middleware;
pub mod persist;
pub mod query_cache;
pub mod selector;
pub mod signal;
pub mod store;
pub mod store_redux;
pub mod undo_manager;

// ── Calendar & Scheduling ──
pub mod calendar;
pub mod cron_expr;
pub mod date_picker;
pub mod duration;
pub mod recurrence;
pub mod relative_time;
pub mod scheduler;
pub mod timezone;

// ── Data Grid & Tables ──
pub mod cell_editor;
pub mod column_config;
pub mod data_export;
pub mod data_grid;
pub mod filter_engine;
pub mod pivot_table;
pub mod row_selection;
pub mod sort_engine;

// ── Math & Science ──
pub mod bezier;
pub mod complex_num;
pub mod fraction;
pub mod interpolation;
pub mod linear_algebra;
pub mod noise;
pub mod random;
pub mod statistics;

// ── Math & Geometry ──
pub mod bigint;
pub mod geometry2d;
pub mod geometry3d;
pub mod matrix_ops;
pub mod polynomial;
pub mod signal_processing;
pub mod stats_engine;

// ── Advanced Graphics ──
pub mod camera2d;
pub mod collision2d;
pub mod physics2d;
pub mod shader;
pub mod sprite;
pub mod svg_path;
pub mod tilemap;
pub mod webgpu;

// ── Geometry & Physics Simulation ──
pub mod curve_math;
pub mod particle_system;
pub mod ray_cast;
pub mod rigid_body;
pub mod spatial_hash;
pub mod spring_physics;

// ── Communication & Social ──
pub mod activity_feed;
pub mod comment;
pub mod email_template;
pub mod notification_center;
pub mod presence;
pub mod reaction;
pub mod share;
pub mod sms;

// ── Compression & Encoding ──
pub mod ascii85;
pub mod base_encoding;
pub mod deflate;
pub mod hex;
pub mod huffman;
pub mod lz77;
pub mod percent_encode;
pub mod url_encode;

// ── Platform ──
pub mod energy;
pub mod storage;

// ── Networking & HTTP ──
pub mod cookie_jar;
pub mod cors;
pub mod dns_resolve;
pub mod grpc_web;
pub mod http2;
pub mod http_client;
pub mod proxy;
pub mod sse_client;

// ── Form Validation & Utilities ──
pub mod debounce;
pub mod field_array;
pub mod form_wizard;
pub mod input_mask;
pub mod json_schema_validate;
pub mod memoize;
pub mod throttle;
pub mod validator;

// ── Accessibility ──
pub mod aria;
pub mod color_contrast;
pub mod focus_trap;
pub mod keyboard_nav;
pub mod live_region;
pub mod reduced_motion;
pub mod screen_reader;
pub mod skip_link;

// ── Canvas & Geometry ──
pub mod canvas_composite;
pub mod canvas_text;
pub mod convex_hull;
pub mod path_boolean;
pub mod polygon_clip;
pub mod svg_animate;
pub mod svg_filter;
pub mod triangulation;

// ── Pattern & String Processing ──
pub mod base64_codec;
pub mod crc_hash;
pub mod fuzzy_match;
pub mod glob_match;
pub mod murmur_hash;
pub mod regex_engine;
pub mod string_distance;
pub mod template_literal;

// ── Concurrency & Scheduling ──
pub mod backpressure;
pub mod channel;
pub mod circuit_breaker;
pub mod event_loop;
pub mod priority_queue;
pub mod scheduler_coop;
pub mod semaphore;
pub mod task_queue;

// ── UI Components ──
pub mod avatar;
pub mod badge;
pub mod chip;
pub mod dropdown;
pub mod modal;
pub mod popover;
pub mod stepper;
pub mod tooltip;

// ── Internationalization & Formatting ──
pub mod bidi_algorithm;
pub mod collation;
pub mod date_format;
pub mod icu_message;
pub mod list_format;
pub mod number_format;
pub mod plural_rules;
pub mod segmenter;

// ── Data Visualization ──
pub mod box_plot;
pub mod bullet_chart;
pub mod heatmap;
pub mod histogram;
pub mod sparkline;
pub mod timeline;
pub mod treemap;
pub mod waterfall;

// ── Design System ──
pub mod breakpoint;
pub mod color_palette;
pub mod container_query;
pub mod design_token;
pub mod grid_system;
pub mod media_query;
pub mod responsive;
pub mod spacing;

// ── Logging & Observability ──
pub mod alerting;
pub mod audit_log;
pub mod distributed_trace;
pub mod feature_metric;
pub mod health_check;
pub mod metrics;
pub mod sli_slo;
pub mod structured_log;

// ── Serialization & Encoding ──
pub mod asn1;
pub mod bencode;
pub mod bson;
pub mod cbor;
pub mod msgpack;
pub mod protobuf;
pub mod toml_parse;
pub mod yaml_parse;

// ── ECS & Game Patterns ──
pub mod behavior_tree;
pub mod ecs;
pub mod game_loop;
pub mod nav_mesh;
pub mod octree;
pub mod pathfinding;
pub mod quadtree;
pub mod steering;

// ── Cryptography & Security ──
pub mod aes;
pub mod chacha20;
pub mod csp;
pub mod csrf;
pub mod pbkdf2;
pub mod sha256;
pub mod sha512;
pub mod xss_protect;

// ── Cryptographic Protocols & Primitives ──
pub mod bloom_crypt;
pub mod commitment;
pub mod hmac_engine;
pub mod merkle_tree;
pub mod oblivious_transfer;
pub mod secret_sharing;
pub mod threshold_crypto;
pub mod zero_knowledge;

// ── Security & Auth ──
pub mod cors_handler;
pub mod csrf_protect;
pub mod input_sanitizer;
pub mod jwt_lite;
pub mod password_hash;
pub mod permission_engine;
pub mod rate_limiter;

// ── CLI & Terminal ──
pub mod ansi_color;
pub mod arg_parse;
pub mod progress_bar;
pub mod prompt;
pub mod spinner;
pub mod table_render;
pub mod term_emulator;
pub mod terminal;

// ── File Formats & Parsing ──
pub mod cron_parse;
pub mod csv_parse;
pub mod ini_parse;
pub mod markdown_parse;
pub mod mime_type;
pub mod semver_parse;
pub mod xml_parse;

// ── Data Structures ──
pub mod bloom_filter;
pub mod disjoint_set;
pub mod interval_tree;
pub mod lru_cache;
pub mod ring_buffer;
pub mod rope;
pub mod skip_list;
pub mod trie;

// ── Advanced Data Structures ──
pub mod avl_tree;
pub mod fenwick_tree;
pub mod red_black_tree;
pub mod segment_tree;
pub mod splay_tree;
pub mod suffix_array;
pub mod treap;
pub mod wavelet_tree;

// ── Image Processing ──
pub mod color_space;
pub mod connected_component;
pub mod dither;
pub mod histogram_image;
pub mod image_blend;
pub mod image_filter;
pub mod morphology;
pub mod pixel_buffer;

// ── Network Protocols ──
pub mod dns_packet;
pub mod http_parse;
pub mod json_rpc;
pub mod mqtt;
pub mod redis_protocol;
pub mod smtp;
pub mod uri_parse;
pub mod websocket_frame;

// ── Template & Document Generation ──
pub mod email_builder;
pub mod handlebars_engine;
pub mod html_builder;
pub mod liquid_template;
pub mod mustache;
pub mod pdf_layout;
pub mod svg_builder;
pub mod text_diff_engine;

// ── Workflow & Automation ──
pub mod cqrs;
pub mod pipeline;
pub mod rule_engine;
pub mod saga;
pub mod scheduler_job;
pub mod state_machine;
pub mod webhook;
pub mod workflow;

// ── Diff, Patch & Merge ──
pub mod change_set;
pub mod crdt_text;
pub mod json_diff;
pub mod myers_diff;
pub mod operational_transform;
pub mod semantic_diff;
pub mod three_way_merge;
pub mod unified_diff;

// ── Queue & Messaging ──
pub mod command_bus;
pub mod event_bus;
pub mod job_queue;
pub mod message_broker;
pub mod message_queue;
pub mod outbox;
pub mod pub_sub;
pub mod stream_buffer;

// ── Configuration & Feature Flags ──
pub mod ab_test;
pub mod config_loader;
pub mod config_merge;
pub mod config_watch;
pub mod env_parse;
pub mod secret_manager;

// ── Search & Indexing ──
pub mod autocomplete_engine;
pub mod bm25;
pub mod faceted_search;
pub mod inverted_index;
pub mod search_highlight;
pub mod search_query_parse;
pub mod spell_checker;
pub mod text_tokenizer;

// ── Process & Runtime Patterns ──
pub mod actor_model;
pub mod coroutine;
pub mod dependency_injector;
pub mod future_combinator;
pub mod plugin_system;
pub mod scheduler_round_robin;
pub mod supervisor;
pub mod worker_pool;

// ── API & GraphQL ──
pub mod api_rate_limit;
pub mod api_versioning;
pub mod content_negotiation;
pub mod graphql_execute;
pub mod graphql_parse;
pub mod openapi_spec;
pub mod rest_resource;
pub mod webhook_dispatch;

// ── ML & Numerical Methods ──
pub mod decision_tree;
pub mod dimensionality;
pub mod kmeans;
pub mod knn;
pub mod naive_bayes;
pub mod neural_net;
pub mod optimization;

// ── Network & Transport ──
pub mod circuit_breaker_adv;
pub mod ip_addr_util;
pub mod load_balancer;
pub mod nat_traversal;
pub mod packet_codec;
pub mod protocol_mux;
pub mod service_registry;
pub mod tcp_state;

// ── Scheduling & Time Management ──
pub mod deadline;
pub mod rate_controller;
pub mod recurring_task;
pub mod retry_policy;
pub mod task_dag;
pub mod temporal_index;
pub mod time_series;
pub mod timer_wheel;

// ── Compiler & Language Tools ──
pub mod bytecode_vm;
pub mod cfg_grammar;
pub mod ir_builder;
pub mod lexer;
pub mod pratt_parser;
pub mod regex_compile;
pub mod symbol_table;
pub mod type_checker;

// ── Encoding & Advanced Formats ──
pub mod bitstream;
pub mod escaped_string;
pub mod hex_dump;
pub mod rle_encoding;
pub mod unicode_util;
pub mod varint;

// ── Graph Algorithms & Distributed Patterns ──
pub mod consensus;
pub mod consistent_hash;
pub mod gossip_protocol;
pub mod graph;
pub mod min_spanning;
pub mod network_flow;
pub mod shortest_path;
pub mod vector_clock;

// ── Distributed Systems ──
pub mod crdt_counter;
pub mod crdt_register;
pub mod crdt_set;
pub mod lamport_clock;
pub mod membership;
pub mod paxos;
pub mod raft_log;
pub mod sharding;

// ── Functional Programming Patterns ──
pub mod functor;
pub mod lazy_eval;
pub mod lens;
pub mod monad;
pub mod pattern_match;
pub mod persistent_ds;
pub mod pipe_compose;
pub mod transducer;

// ── OS & System Patterns ──
pub mod disk_scheduler;
pub mod file_system_journal;
pub mod ipc_channel;
pub mod memory_allocator;
pub mod page_replacement;
pub mod process_tree;
pub mod resource_manager;
pub mod virtual_fs;

// ── Domain-Driven Design & Modeling ──
pub mod aggregate;
pub mod bounded_context;
pub mod domain_event;
pub mod entity;
pub mod money;
pub mod repository_pattern;
pub mod specification;
pub mod value_object;

// ── Content & Media Processing ──
pub mod content_pipeline;
pub mod feed_generator;
pub mod image_metadata;
pub mod markdown_ext;
pub mod media_playlist;
pub mod sitemap_builder;
pub mod slug_generator;
pub mod syntax_highlight;

// ── Monitoring & Operations ──
pub mod anomaly_detect;
pub mod canary_deploy;
pub mod capacity_planner;
pub mod chaos_engine;
pub mod incident_tracker;
pub mod log_aggregator;
pub mod trace_context;
pub mod uptime_monitor;

// ── Data Pipeline & ETL ──
pub mod batch_processor;
pub mod data_pipeline_etl;
pub mod data_quality;
pub mod data_sampler;
pub mod data_transform;
pub mod data_validator;
pub mod dedup_engine;
pub mod schema_evolution;

// ── WebAssembly & Runtime Concepts ──
pub mod capability;
pub mod gc_collector;
pub mod jit_concept;
pub mod module_loader;
pub mod sandbox;
pub mod wasm_interp;
pub mod wasm_memory;
pub mod wasm_module;

// ── UI Framework Core ──
pub mod component_lifecycle;
pub mod event_delegation;
pub mod hydration;
pub mod reconciler;
pub mod render_tree;
pub mod style_resolver;
pub mod vdom_diff;

// ── NLP & Text Analysis ──
pub mod ner_tagger;
pub mod sentiment;
pub mod stemmer;
pub mod stop_words;
pub mod text_classify;
pub mod text_summarize;
pub mod tfidf;
pub mod tokenize_nlp;

// ── Blockchain & Ledger Concepts ──
pub mod blockchain;
pub mod consensus_pow;
pub mod ledger;
pub mod nft_registry;
pub mod smart_contract;
pub mod state_trie;
pub mod token_ledger;
pub mod tx_pool;

// ── Metrics & Telemetry ──
pub mod histogram_metrics;
pub mod meter;
pub mod metric_export;
pub mod metric_registry;
pub mod prometheus_fmt;
pub mod reservoir;
pub mod resource_metrics;
pub mod span_metrics;

// ── Compiler Backend & Code Generation ──
pub mod register_alloc;

// ── CLI & Developer Tools ──
pub mod changelog;
pub mod code_formatter;
pub mod codegen_emit;
pub mod dep_graph;
pub mod doc_generator;
pub mod dominance;
pub mod instruction_select;
pub mod linker;
pub mod linter_engine;
pub mod liveness;
pub mod optimizer_pass;
pub mod peephole;
pub mod repl;
pub mod scaffold;
pub mod task_runner;

// ── Advanced Networking ──
pub mod api_pagination;
pub mod http_cache;
pub mod http_middleware;
pub mod http_router;
pub mod oauth_flow;
pub mod request_dedup;
pub mod sse_protocol;
pub mod webhook_verify;

// ── Advanced Visualization & Charts ──
pub mod calendar_heatmap;
pub mod chord_diagram;
pub mod density_plot;
pub mod funnel_chart;
pub mod gauge_chart;
pub mod network_graph;
pub mod sankey_diagram;

// ── Error Handling & Resilience ──
pub mod bulkhead;
pub mod compensation;
pub mod error_budget;
pub mod error_chain;
pub mod fallback;
pub mod graceful_degradation;
pub mod hedge_request;
pub mod timeout_manager;

// ── IoT & Embedded Patterns ──
pub mod device_registry;
pub mod edge_compute;
pub mod gpio_sim;
pub mod i2c_sim;
pub mod ota_update;
pub mod power_mgmt;
pub mod sensor_fusion;
pub mod telemetry_ingest;

// ── Type System & Generic Abstractions ──
pub mod builder_derive;
pub mod generic_collection;
pub mod newtype;
pub mod phantom_types;
pub mod result_ext;
pub mod schema_type;
pub mod tagged_union;
pub mod type_registry;

// ── Classic Design Patterns ──
pub mod chain_of_responsibility;
pub mod flyweight;
pub mod interpreter;
pub mod mediator;
pub mod memento;
pub mod object_mapper;
pub mod proxy_pattern;
pub mod strategy_pattern;

// ── Simulation & Modeling ──
pub mod agent_sim;
pub mod cellular_automaton;
pub mod epidemic_sim;
pub mod fluid_sim;
pub mod markov_chain;
pub mod monte_carlo;
pub mod petri_net;
pub mod queuing_sim;

// ── Event Sourcing & CQRS Extended ──
pub mod command_handler;
pub mod event_migration;
pub mod event_store;
pub mod event_subscription;
pub mod process_manager;
pub mod projection;
pub mod read_model;
pub mod snapshot_store;

// ── Access Control & Policy ──
pub mod abac;
pub mod access_log;
pub mod api_key_mgr;
pub mod data_masking;
pub mod ip_filter;
pub mod policy_engine;
pub mod session_mgr;
pub mod tenant_isolation;

// ── Protocol Buffers & RPC ──
pub mod proto_decode;
pub mod proto_encode;
pub mod proto_reflect;
pub mod proto_schema;
pub mod rpc_discovery;
pub mod rpc_framework;
pub mod rpc_middleware;
pub mod rpc_testing;

// ── RPC & Wire Protocol ──
pub mod binary_rpc;
pub mod rpc_batch;
pub mod rpc_codec;
pub mod rpc_intercept;
pub mod rpc_retry;
pub mod rpc_stream;
pub mod service_stub;
pub mod wire_format;

// ── Numerical Computing ──
pub mod finite_diff;
pub mod ode_solver;
pub mod quadrature;
pub mod random_dist;
pub mod root_finding;
pub mod sparse_matrix;

// ── File System & Storage ──
pub mod block_device;
pub mod compaction;
pub mod extent_alloc;
pub mod file_index;
pub mod lsm_tree;
pub mod sstable;
pub mod versioned_store;
pub mod wal_manager;

// ── Workflow & Automation 2 ──
pub mod approval_workflow;
pub mod batch_scheduler;
pub mod form_engine;
pub mod integration_hub;
pub mod notification_engine;
pub mod report_builder;
pub mod template_processor;

// ── Auth & Security Web ──
pub mod csrf_protection;
pub mod oauth2_flow;
pub mod oidc;
pub mod security_headers;

// ── API & Transport ──
pub mod etag;
pub mod graphql_schema;
pub mod http2_frame;
pub mod jsonrpc;
pub mod multipart;

// ── i18n & Content ──
pub mod feed_gen;
pub mod image_ops;
pub mod l10n;
pub mod pdf_gen;
pub mod qr_code;
pub mod sitemap;

// ── Modern Frontend Patterns ──
pub mod ab_testing;
pub mod css_in_rs;
pub mod pwa_manifest;
pub mod seo;
pub mod service_worker;

// ── Data & Validation ──
pub mod csv_engine;
pub mod json_pointer;
pub mod semver;

// ── Infra & DevOps ──
pub mod blue_green;
pub mod chaos_eng;
pub mod feature_gate;

// ── Realtime & Messaging ──
pub mod channel_mux;
pub mod collab_edit;
pub mod live_cursor;
pub mod long_poll;
pub mod push_notify;
pub mod webrtc_signal;

// ── Testing & DX ──
pub mod contract_test;
pub mod error_page;
pub mod fixture_factory;
pub mod http_mock;
pub mod load_profile;

// ── Production Ops ──
pub mod compression_negotiate;
pub mod graceful_shutdown;
pub mod request_coalesce;
pub mod request_id;
pub mod response_cache;

// ── Observability & Monitoring ──
pub mod audit_trail;
pub mod benchmark_harness;
pub mod cost_tracker;
pub mod sla_tracker;

// ── Data Pipeline ──
pub mod change_data_capture;
pub mod data_pipeline;
pub mod dead_letter;
pub mod idempotency;
pub mod saga_orchestrator;
pub mod schema_registry;
pub mod stream_window;

// ── API Gateway Patterns ──
pub mod api_gateway;
pub mod jwt_gateway;
pub mod request_transform;
pub mod response_transform;
pub mod service_mesh;
pub mod traffic_mirror;

// ── Developer Tooling ──
pub mod code_gen;
pub mod debug_toolbar;
pub mod dependency_check;
pub mod env_validator;
pub mod migration_runner;
pub mod openapi_client;

// ── State & Caching ──
pub mod cache_aside;
pub mod content_store;
pub mod distributed_lock;
pub mod pagination;
pub mod search_index;

// ── Auth & Identity ──
pub mod invite_system;
pub mod mfa;
pub mod org_hierarchy;
pub mod permission_cache;
pub mod rbac;
pub mod social_auth;
pub mod token_bucket_auth;
pub mod user_profile;

// ── Media & Encoding ──
pub mod canvas_2d;
pub mod content_disposition;
pub mod data_uri;
pub mod media_type;
pub mod sprite_sheet;

// ── Game Engine ECS ──
pub mod archetype_storage;
pub mod change_detection_ecs;
pub mod component_registry;
pub mod entity_commands;
pub mod entity_store;
pub mod query_engine_ecs;
pub mod system_scheduler;
pub mod world_state;

// ── Game Input & Mapping ──
pub mod action_mapping;
pub mod cursor_lock;
pub mod gamepad_input;
pub mod haptic_feedback;
pub mod input_manager;
pub mod input_replay;
pub mod key_binding_game;
pub mod touch_gesture_game;

// ── Game Loop & Scene Management ──
pub mod camera_system;
pub mod draw_call_batcher;
pub mod frame_graph;
pub mod game_tick_loop;
pub mod render_queue;
pub mod scene_graph_3d;
pub mod scene_manager;
pub mod viewport_manager;

// ── Game AI & Navigation ──
pub mod behavior_tree_ai;
pub mod flocking_sim;
pub mod flow_field_path;
pub mod formation;
pub mod navmesh;
pub mod pathfind_astar;
pub mod steering_behavior;
pub mod utility_ai;

// ── 2D Physics Engine ──
pub mod broadphase_2d;
pub mod collision_2d;
pub mod contact_solver_2d;
pub mod joint_2d;
pub mod narrowphase_2d;
pub mod physics_world_2d;
pub mod rigid_body_2d;
pub mod spatial_grid_2d;

// ── 3D Physics Engine ──
pub mod broadphase_3d;
pub mod collision_3d;
pub mod constraint_solver_3d;
pub mod contact_manifold;
pub mod gjk_epa;
pub mod ragdoll_physics;
pub mod rigid_body_3d;
pub mod vehicle_physics;

// ── 3D Rendering Fundamentals ──
pub mod beam_renderer;
pub mod bounding_volume;
pub mod decal_projector;
pub mod distortion_fx;
pub mod frustum_cull;
pub mod gpu_particles;
pub mod index_buffer;
pub mod instanced_draw;
pub mod lod_system;
pub mod mesh_builder;
pub mod particle_affector;
pub mod particle_emitter;
pub mod ribbon_trail;
pub mod transform_hierarchy;
pub mod vertex_format;
pub mod volumetric_fog;

// ── PBR Materials & Shaders ──
pub mod anisotropic_material;
pub mod clearcoat;
pub mod normal_mapping;
pub mod parallax_mapping;
pub mod pbr_material;
pub mod shader_graph;
pub mod subsurface_scatter;
pub mod texture_sampler;

// ── Lighting Engine ──
pub mod area_light;
pub mod directional_light;
pub mod irradiance_volume;
pub mod light_probe;
pub mod point_light;
pub mod shadow_atlas;
pub mod shadow_cascade;
pub mod spot_light;

// ── Post-Processing Effects ──
pub mod bloom_fx;
pub mod chromatic_aberration;
pub mod color_grading_render;
pub mod depth_of_field;
pub mod motion_blur;
pub mod screen_space_reflect;
pub mod ssao;
pub mod tone_mapper;

// ── Rendering Quality & Anti-Aliasing ──
pub mod aniso_filter;
pub mod dithering;
pub mod fxaa;
pub mod mipmap_gen;
pub mod msaa_resolve;
pub mod sharpen_filter;
pub mod taa;
pub mod upscaler;

// ── Terrain & Environment ──
pub mod cloud_renderer;
pub mod grass_renderer;
pub mod heightmap_terrain;
pub mod sky_atmosphere;
pub mod terrain_splat;
pub mod tree_billboard;
pub mod water_surface;
pub mod weather_system;

// ── Text & UI Rendering ──
pub mod glyph_cache;
pub mod msdf_text;
pub mod nine_slice;
pub mod sdf_font;
pub mod signed_distance_2d;
pub mod text_layout_engine;
pub mod ui_renderer;
pub mod vector_rasterizer;

// ── Ray Tracing ──
pub mod brdf_model;
pub mod bvh_builder;
pub mod denoiser;
pub mod importance_sample;
pub mod monte_carlo_render;
pub mod path_tracer;
pub mod photon_map;
pub mod ray_intersect;

// ── Audio Engine ──
pub mod audio_bus;
pub mod audio_graph_engine;
pub mod audio_mixer;
pub mod audio_pool;
pub mod crossfade_audio;
pub mod hrtf;
pub mod reverb_engine;
pub mod spatial_audio_3d;

// ── Audio Synthesis ──
pub mod audio_filter_dsp;
pub mod envelope_adsr;
pub mod fm_synthesis;
pub mod granular_synth;
pub mod noise_gen_audio;
pub mod oscillator_synth;
pub mod vocoder;
pub mod wavetable;

// ── Music Systems ──
pub mod arpeggiator;
pub mod chord_progression;
pub mod midi_parser;
pub mod music_theory;
pub mod procedural_music;
pub mod scale_quantize;
pub mod sequencer;
pub mod tempo_sync;

// ── Procedural Terrain ──
pub mod biome_generator;
pub mod cave_generator;
pub mod erosion_sim;
pub mod fractal_brownian;
pub mod perlin_noise;
pub mod river_network;
pub mod simplex_noise;
pub mod worley_noise;

// ── Procedural Content ──
pub mod corridor_linker;
pub mod dungeon_generator;
pub mod l_system;
pub mod loot_table;
pub mod maze_builder;
pub mod name_generator;
pub mod room_placer;
pub mod texture_synth;

// ── Procedural Geometry ──
pub mod catmull_clark;
pub mod extrude_path;
pub mod lathe_mesh;
pub mod loop_subdivision;
pub mod mesh_boolean;
pub mod mesh_simplify;
pub mod parametric_surface;
pub mod subdivision_surface;

// ── World Streaming & Assets ──
pub mod asset_loader;
pub mod chunk_manager;
pub mod hot_reload_assets;
pub mod infinite_terrain;
pub mod level_streaming;
pub mod object_pool_game;
pub mod prefab_system;
pub mod resource_cache_game;

// ── Game Systems ──
pub mod achievement_system;
pub mod crafting_system;
pub mod dialogue_tree;
pub mod inventory_system;
pub mod leaderboard;
pub mod quest_tracker;
pub mod replay_recorder;
pub mod save_game;

// ── Fluid Dynamics ──
pub mod eulerian_fluid;
pub mod fluid_coupling;
pub mod fluid_surface;
pub mod navier_stokes;
pub mod pressure_solver;
pub mod sph_fluid;
pub mod velocity_field;
pub mod vortex_particle;

// ── Soft Body Simulation ──
pub mod cloth_sim;
pub mod deformable_mesh;
pub mod mass_spring;
pub mod position_based_dynamics;
pub mod rope_physics;
pub mod shape_matching;
pub mod tear_sim;
pub mod volume_preserve;

// ── N-Body & Orbital Mechanics ──
pub mod barnes_hut;
pub mod celestial_render;
pub mod galaxy_gen;
pub mod gravity_field;
pub mod nbody_sim;
pub mod orbital_mechanics;
pub mod star_catalog;
pub mod trajectory_planner;

// ── Cellular Automata & Swarm ──
pub mod ant_colony;
pub mod boids_advanced;
pub mod cellular_automaton_2d;
pub mod crowd_sim;
pub mod game_of_life;
pub mod langton_ant;
pub mod reaction_diffusion;
pub mod swarm_intelligence;

// ── Optimization Algorithms ──
pub mod bayesian_opt;
pub mod constraint_satisfy;
pub mod evolutionary_strategy;
pub mod genetic_algorithm;
pub mod hill_climbing;
pub mod multi_objective;
pub mod particle_swarm;
pub mod simulated_annealing;

// ── Signal Processing ──
pub mod bandpass_filter;
pub mod convolution_engine;
pub mod fft_engine;
pub mod fir_filter;
pub mod iir_filter;
pub mod mel_spectrogram;
pub mod stft;
pub mod window_function;

// ── Control Systems ──
pub mod adaptive_control;
pub mod complementary_filter;
pub mod extended_kalman;
pub mod feedforward;
pub mod kalman_filter;
pub mod model_predictive;
pub mod pid_controller;
pub mod state_observer;

// ── Numerical Linear Algebra ──
pub mod conjugate_gradient;
pub mod eigenvalue_solver;
pub mod least_squares;
pub mod lu_decompose;
pub mod polynomial_fit;
pub mod qr_decompose;
pub mod sparse_matrix_solver;
pub mod svd_decompose;

// ── Peer-to-Peer ──
pub mod dht_store;
pub mod hole_punch;
pub mod mesh_topology;
pub mod p2p_relay;
pub mod peer_auth;
pub mod peer_discovery;
pub mod peer_score;
pub mod relay_server;

// ── Lobby & Matchmaking ──
pub mod lobby_system;
pub mod matchmaker;
pub mod matchmaker_elo;
pub mod queue_match;
pub mod region_select;
pub mod room_manager_net;
pub mod session_relay;
pub mod team_balance;

// ── Realtime Sync ──
pub mod authority_model;
pub mod bandwidth_throttle;
pub mod clock_sync;
pub mod conflict_resolve_net;
pub mod entity_replication;
pub mod eventual_consist;
pub mod interest_mgmt;
pub mod state_sync;

// ── Netcode ──
pub mod delta_compress_net;
pub mod jitter_buffer_net;
pub mod lag_compensate;
pub mod netcode_predict;
pub mod netcode_rollback;
pub mod netcode_snapshot;
pub mod packet_loss_sim;
pub mod tick_sync;

// ── Chat & Social ──
pub mod chat_history;
pub mod chat_room;
pub mod emote_system;
pub mod friend_list;
pub mod guild_system;
pub mod message_dispatch;
pub mod online_status;
pub mod voice_lobby;

// ── Transport Layer ──
pub mod congestion_ctrl;
pub mod connection_migrate;
pub mod flow_control_net;
pub mod keepalive_monitor;
pub mod quic_transport;
pub mod transport_encrypt;
pub mod transport_mux;
pub mod udp_reliable;

// ── Anti-Cheat ──
pub mod aimbot_detect;
pub mod anti_cheat;
pub mod ban_system;
pub mod cheat_heuristic;
pub mod input_validate_net;
pub mod replay_validate;
pub mod server_authority;
pub mod speed_hack_detect;

// ── Neural Net Layers ──
pub mod activation_fn;
pub mod attention_layer;
pub mod batch_norm;
pub mod conv2d_layer;
pub mod dense_layer;
pub mod dropout_layer;
pub mod embedding_layer_nn;
pub mod pooling_layer;

// ── Training & Optimization ──
pub mod adam_optimizer;
pub mod backprop_engine;
pub mod early_stopping;
pub mod gradient_descent;
pub mod loss_function;
pub mod lr_scheduler;
pub mod regularization;
pub mod weight_init;

// ── Transformer Architecture ──
pub mod beam_search;
pub mod feed_forward_block;
pub mod layer_norm_transformer;
pub mod multi_head_attn;
pub mod positional_encode;
pub mod self_attention;
pub mod transformer_decoder;
pub mod transformer_encoder;

// ── Inference Engine ──
pub mod batch_inference;
pub mod knowledge_distill;
pub mod model_cache_infer;
pub mod model_runtime;
pub mod onnx_parse;
pub mod pruning;
pub mod quantize_engine;
pub mod tensor_graph;

// ── Computer Vision ──
pub mod camera_calibrate;
pub mod edge_detect_cv;
pub mod feature_detect_cv;
pub mod image_pyramid;
pub mod object_track;
pub mod optical_flow;
pub mod stereo_depth;
pub mod template_match;

// ── Reinforcement Learning ──
pub mod bandit_algo;
pub mod environment_sim;
pub mod multi_agent_rl;
pub mod policy_gradient;
pub mod q_learning;
pub mod replay_buffer_rl;
pub mod reward_shaping;
pub mod td_learning;

// ── ML Data Pipeline ──
pub mod augmentation;
pub mod confusion_matrix;
pub mod cross_validation;
pub mod data_loader_ml;
pub mod feature_engineer;
pub mod normalization_ml;
pub mod precision_recall;
pub mod roc_curve;

// ── Generative Models ──
pub mod autoencoder;
pub mod diffusion_model;
pub mod flow_model;
pub mod gan_engine;
pub mod image_generate;
pub mod latent_space;
pub mod style_transfer_nn;
pub mod variational_ae;

// ── Kinematics ──
pub mod dh_parameters;
pub mod forward_kinematics;
pub mod inverse_kinematics;
pub mod joint_space;
pub mod motion_profile;
pub mod serial_chain;
pub mod trajectory_gen;
pub mod workspace_analysis;

// ── Motion Planning ──
pub mod a_star_3d;
pub mod dynamic_window;
pub mod lattice_planner;
pub mod motion_primitive;
pub mod potential_field;
pub mod prm_planner;
pub mod rrt_planner;
pub mod velocity_obstacle;

// ── SLAM & Mapping ──
pub mod ekf_slam;
pub mod icp_match;
pub mod loop_closure;
pub mod occupancy_grid;
pub mod particle_filter_slam;
pub mod point_cloud_proc;
pub mod pose_graph;
pub mod visual_odometry;

// ── Perception & Sensors ──
pub mod depth_camera;
pub mod encoder_sensor;
pub mod gps_filter;
pub mod imu_fusion;
pub mod lidar_process;
pub mod radar_process;
pub mod sensor_model;
pub mod sonar_range;

// ── Control & Actuation ──
pub mod admittance_control;
pub mod force_torque;
pub mod hybrid_force_pos;
pub mod impedance_control;
pub mod motor_control;
pub mod pwm_control;
pub mod servo_driver;
pub mod stepper_control;

// ── Autonomous Navigation ──
pub mod behavior_plan;
pub mod global_planner;
pub mod lane_detect;
pub mod local_planner;
pub mod mission_planner;
pub mod obstacle_avoid;
pub mod traffic_sign;
pub mod waypoint_follow;

// ── Manipulation ──
pub mod assembly_plan;
pub mod bin_picking;
pub mod compliant_grasp;
pub mod deformable_manip;
pub mod grasp_planner;
pub mod pick_place;
pub mod push_manip;
pub mod tool_use_robot;

// ── Multi-Robot Systems ──
pub mod auction_allocate;
pub mod comm_robot;
pub mod consensus_robot;
pub mod coverage_plan;
pub mod fleet_manage;
pub mod formation_control;
pub mod swarm_robot;
pub mod task_allocation;

// ── W26 Bioinformatics: Sequence Analysis ──────────────────────────
pub mod blast_search;
pub mod compressed_sa;
pub mod consensus_seq;
pub mod kmer_count;
pub mod needleman_wunsch;
pub mod sequence_align;
pub mod smith_waterman;
pub mod wavelet_tree_bio;

// ── W26 Bioinformatics: Phylogenetics ──────────────────────────────
pub mod bootstrap_phylo;
pub mod clade_analysis;
pub mod molecular_clock;
pub mod neighbor_join;
pub mod newick_parse;
pub mod phylo_tree;
pub mod tree_traversal_bio;
pub mod upgma_cluster;

// ── W26 Bioinformatics: Structural Biology ─────────────────────────
pub mod disulfide_bridge;
pub mod hydrogen_bond;
pub mod molecular_surface;
pub mod pdb_format;
pub mod pdb_parse;
pub mod protein_fold;
pub mod ramachandran;
pub mod secondary_structure;
pub mod structural_align;

// ── W26 Bioinformatics: Genomics ───────────────────────────────────
pub mod gene_predict;
pub mod genome_annotation;
pub mod genome_compare;
pub mod promoter_detect;
pub mod read_mapper;
pub mod splice_site;
pub mod variant_call;
pub mod vcf_parse;

// ── W26 Bioinformatics: Proteomics ─────────────────────────────────
pub mod charge_deconv;
pub mod isotope_pattern;
pub mod mass_spectrum;
pub mod peptide_fragment;
pub mod post_translate_mod;
pub mod protein_digest;
pub mod protein_quant;
pub mod spectral_match;

// ── W26 Bioinformatics: Population Genetics ────────────────────────
pub mod allele_freq;
pub mod coalescent_sim;
pub mod effective_pop;
pub mod fst_divergence;
pub mod hardy_weinberg;
pub mod linkage_disequi;
pub mod migration_model;
pub mod selection_test;

// ── W26 Bioinformatics: Bio Formats ───────────────────────────────
pub mod bed_parse;
pub mod codon_table;
pub mod fasta_parse;
pub mod fastq_parse;
pub mod gff_parse;
pub mod sam_parse;

// ── W26 Bioinformatics: Bio Algorithms ────────────────────────────
pub mod bloom_filter_bio;
pub mod burrows_wheeler;
pub mod de_bruijn_graph;
pub mod distance_matrix_bio;
pub mod fm_index;
pub mod min_hash_bio;
pub mod orf_finder;
pub mod sequence_assembly;
pub mod suffix_tree_bio;

// ── W27 Financial: Order Book & Matching ──────────────────────────
pub mod auction_market;
pub mod book_depth;
pub mod crossing_engine;
pub mod dark_pool;
pub mod execution_algo;
pub mod iceberg_order;
pub mod limit_order;
pub mod market_order;
pub mod order_amend;
pub mod order_book;
pub mod order_cancel;
pub mod order_match;
pub mod price_level;
pub mod trade_engine;
pub mod trade_match;

// ── W27 Financial: Market Data ────────────────────────────────────
pub mod candle_aggregate;
pub mod market_depth;
pub mod market_feed;
pub mod market_maker;
pub mod order_flow;
pub mod tick_data;
pub mod tick_replay;
pub mod vwap_calc;

// ── W27 Financial: Risk Management ────────────────────────────────
pub mod credit_risk;
pub mod drawdown_risk;
pub mod exposure_calc;
pub mod monte_carlo_risk;
pub mod risk_budget;
pub mod risk_parity;
pub mod scenario_gen;
pub mod stress_test;
pub mod var_model;

// ── W27 Financial: Portfolio ──────────────────────────────────────
pub mod asset_allocate;
pub mod benchmark_track;
pub mod factor_model;
pub mod performance_attrib;
pub mod portfolio_optimize;
pub mod rebalance_engine;
pub mod tax_lot_track;

// ── W27 Financial: Derivatives & Fixed Income ─────────────────────
pub mod binomial_tree;
pub mod black_scholes;
pub mod bond_price;
pub mod greeks_calc;
pub mod option_strategy;
pub mod swap_value;
pub mod volatility_surface;
pub mod yield_curve;

// ── W27 Financial: Settlement & Compliance ────────────────────────
pub mod clearing_house;
pub mod collateral_calc;
pub mod custody_manage;
pub mod kyc_check;
pub mod margin_calc;
pub mod netting_engine;
pub mod reconcile_trade;
pub mod settlement_engine;

// ── W27 Financial: Surveillance & Analytics ───────────────────────
pub mod anomaly_market;
pub mod audit_trail_fin;
pub mod best_exec_check;
pub mod position_limit;
pub mod regulatory_report;
pub mod signal_detect;
pub mod trade_surveil;
pub mod transaction_monitor;
pub mod twap_algo;
pub mod vwap_algo;
pub mod wash_trade_detect;

// ── W28 GIS: Coordinate Systems ───────────────────────────────────
pub mod coord_convert;
pub mod crs_registry;
pub mod datum_transform;
pub mod geo_coord;
pub mod geoid_model;
pub mod great_circle;
pub mod map_project;
pub mod utm_grid;

// ── W28 GIS: Spatial Indexing ─────────────────────────────────────
pub mod geohash_encode;
pub mod h3_index;
pub mod kd_tree_geo;
pub mod quadtree_geo;
pub mod rtree_index;
pub mod spatial_join;
pub mod tile_index;
pub mod voronoi_geo;

// ── W28 GIS: Vector Operations ────────────────────────────────────
pub mod geo_buffer;
pub mod geo_line;
pub mod geo_measure;
pub mod geo_overlay;
pub mod geo_point;
pub mod geo_polygon;
pub mod geo_topology;
pub mod geo_transform;

// ── W28 GIS: Raster Analysis ─────────────────────────────────────
pub mod dem_analysis;
pub mod interpolate_geo;
pub mod ndvi_index;
pub mod raster_algebra;
pub mod raster_classify;
pub mod raster_grid;
pub mod terrain_analysis;
pub mod watershed_calc;

// ── W28 GIS: Routing & Network ───────────────────────────────────
pub mod catchment_area;
pub mod fleet_route;
pub mod isochrone;
pub mod network_flow_geo;
pub mod route_graph;
pub mod service_area;
pub mod shortest_path_geo;
pub mod turn_restrict;

// ── W28 GIS: Map Rendering ───────────────────────────────────────
pub mod choropleth;
pub mod contour_gen;
pub mod heatmap_geo;
pub mod label_place;
pub mod map_graticule;
pub mod map_legend;
pub mod map_style;
pub mod tile_render;

// ── W28 GIS: Geospatial Formats ──────────────────────────────────
pub mod csv_geo;
pub mod geojson_parse;
pub mod gpx_parse;
pub mod kml_parse;
pub mod mvt_encode;
pub mod shapefile_parse;
pub mod wkb_parse;
pub mod wkt_parse;

// ── W28 GIS: Spatial Analysis ─────────────────────────────────────
pub mod hotspot_analysis;
pub mod minimap;
pub mod point_pattern;
pub mod spatial_autocorr;
pub mod spatial_cluster;
pub mod spatial_regress;

// ── W29 Post-Quantum: Lattice Crypto ──────────────────────────────
pub mod gf_arithmetic;
pub mod key_encaps;
pub mod kyber_kem;
pub mod lattice_reduce;
pub mod lattice_sample;
pub mod lwe_core;
pub mod ntru_encrypt;
pub mod ntt_engine;

// ── W29 Post-Quantum: Hash-Based Signatures ──────────────────────
pub mod hash_tree;
pub mod lms_sign;
pub mod sphincs_sign;
pub mod wots_core;
pub mod xmss_sign;

// ── W29 Post-Quantum: Code-Based ─────────────────────────────────
pub mod bike_kem;
pub mod hqc_kem;
pub mod mceliece_kem;

// ── W29 Post-Quantum: Isogeny ─────────────────────────────────────
pub mod csidh_exchange;
pub mod isogeny_core;
pub mod sike_kem;

// ── W29 Post-Quantum: Multivariate & Signatures ──────────────────
pub mod dilithium_sign;
pub mod multivar_core;
pub mod ov_sign;
pub mod rainbow_sign;

// ── W29 Post-Quantum: PQ Key Exchange ─────────────────────────────
pub mod hybrid_kem;
pub mod kem_combiner;
pub mod pq_handshake;
pub mod pq_tls_helper;

// ── W29 Post-Quantum: PQ Infrastructure ──────────────────────────
pub mod fhe_basic;
pub mod pq_aead;
pub mod pq_benchmark;
pub mod pq_cert;
pub mod pq_envelope;
pub mod pq_hash;
pub mod pq_kdf;
pub mod pq_keystore;
pub mod pq_mac;
pub mod pq_migration;
pub mod pq_pki;
pub mod pq_rng;
pub mod pq_serialize;

// ── W29 Post-Quantum: ZK Proofs ──────────────────────────────────
pub mod groth16_core;
pub mod merkle_proof;
pub mod range_proof;
pub mod sigma_protocol;
pub mod stark_core;
pub mod zk_circuit;
pub mod zk_commit;
pub mod zk_identity;

// ── W30 CAD/CAM: NURBS & Curves ──────────────────────────────────
pub mod bezier_curve;
pub mod bspline_curve;
pub mod conic_section;
pub mod curve_fit;
pub mod curve_intersect;
pub mod curve_offset;
pub mod nurbs_curve;
pub mod nurbs_surface;

// ── W30 CAD/CAM: Surface Modeling ─────────────────────────────────
pub mod loft_surface;
pub mod spline_surface;
pub mod surface_analysis;
pub mod surface_fillet;
pub mod surface_intersect;
pub mod surface_offset;
pub mod surface_trim;
pub mod sweep_surface;

// ── W30 CAD/CAM: Solid Modeling ───────────────────────────────────
pub mod boolean_mesh;
pub mod brep_solid;
pub mod csg_ops;
pub mod extrude_revolve;
pub mod parametric_model;
pub mod shell_offset;
pub mod solid_primitive;
pub mod solid_query;

// ── W30 CAD/CAM: Mesh Operations ─────────────────────────────────
pub mod mesh_curvature;
pub mod mesh_gen;
pub mod mesh_parameterize;
pub mod mesh_repair;
pub mod mesh_smooth;
pub mod mesh_subdivide;

// ── W30 CAD/CAM: Tolerancing & GD&T ──────────────────────────────
pub mod datum_ref;
pub mod fit_system;
pub mod gdt_tolerance;
pub mod position_tol;
pub mod runout_tol;
pub mod stack_analysis;
pub mod surface_finish;

// ── W30 CAD/CAM: Tool Path ───────────────────────────────────────
pub mod cut_simulate;
pub mod gcode_gen;
pub mod inspection_plan;
pub mod post_process;
pub mod tool_library;
pub mod toolpath_3axis;
pub mod toolpath_contour;
pub mod toolpath_drill;
pub mod toolpath_pocket;

// ── W30 CAD/CAM: Assembly & Formats ──────────────────────────────
pub mod assembly_constraint;
pub mod assembly_tree;
pub mod constraint_solve;
pub mod iges_parse;
pub mod obj_parse;
pub mod sig_parse;
pub mod step_parse;
pub mod stl_parse;

// ── W31 Healthcare: Clinical Data ─────────────────────────────────
pub mod care_plan;
pub mod clinical_note;
pub mod clinical_pathway;
pub mod ehr_record;
pub mod fhir_resource;
pub mod hl7_parse;
pub mod order_set;
pub mod patient_timeline;

// ── W31 Healthcare: Medical Coding ────────────────────────────────
pub mod cda_document;
pub mod code_crosswalk;
pub mod cpt_code;
pub mod drg_assign;
pub mod icd_code;
pub mod loinc_code;
pub mod ndc_lookup;
pub mod snomed_concept;

// ── W31 Healthcare: Clinical Decision ─────────────────────────────
pub mod allergy_check;
pub mod cds_rule;
pub mod dose_calc;
pub mod drug_interact;
pub mod evidence_grade;
pub mod risk_score;
pub mod risk_stratify;
pub mod screening_eval;

// ── W31 Healthcare: Medical Imaging ───────────────────────────────
pub mod dicom_parse;
pub mod dicom_pixel;
pub mod image_register;
pub mod image_window;
pub mod pacs_query;
pub mod path_report;
pub mod roi_measure;
pub mod series_sort;

// ── W31 Healthcare: Population Health ─────────────────────────────
pub mod cohort_study;
pub mod epi_rate;
pub mod quality_measure;
pub mod registries;
pub mod sir_model;
pub mod survival_analysis;

// ── W31 Healthcare: Lab & Diagnostics ─────────────────────────────
pub mod blood_gas;
pub mod coag_monitor;
pub mod lab_panel;
pub mod lab_result;
pub mod micro_result;
pub mod qc_westgard;
pub mod urinalysis;
pub mod vital_signs;

// ── W31 Healthcare: Pharmacy ──────────────────────────────────────
pub mod dispense_calc;
pub mod formulary;
pub mod iv_calc;
pub mod med_admin;
pub mod med_error;
pub mod med_reconcile;
pub mod pk_model;
pub mod rxnorm_drug;

// ── W31 Healthcare: Privacy & Compliance ──────────────────────────
pub mod access_audit;
pub mod breach_assess;
pub mod compliance_check;
pub mod consent_manage;
pub mod data_retention;
pub mod hipaa_deident;
pub mod phi_detect;
pub mod role_access;

// ── W31 Healthcare: Imaging (additional) ──────────────────────────
pub mod annotation_layer;

// ── W28 GIS: Geocoding & Symbology (additional) ──────────────────
pub mod geocode_engine;
pub mod symbol_render;

// ── W30 CAD/CAM: Solid Ops (additional) ──────────────────────────
pub mod chamfer_blend;

// ── Game Engine (additional) ──────────────────────────────────────
pub mod hud_layout;
pub mod parallax_scroll;
pub mod pixel_perfect;
pub mod screen_shake;
pub mod sprite_animator;
pub mod tilemap_engine;
pub mod transition_fx;

// ── Dispatch Registry ───────────────────────────────────────────
pub mod dispatch;
