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
pub mod vdom;
pub mod reactive;
pub mod component;
pub mod html;

// ── Application Logic ──
pub mod router;
pub mod state;
pub mod forms;
pub mod fetch;

// ── UI & Presentation ──
pub mod css;
pub mod animation;
pub mod a11y;
pub mod i18n;

// ── Interaction ──
pub mod drag;
pub mod gesture;
pub mod hotkey;

// ── Data & Display ──
pub mod virtualize;
pub mod table;
pub mod chart;
pub mod surface3d;
pub mod stat_chart;
pub mod clustermap;
pub mod facet;
pub mod stat_overlay;
pub mod highdim;
pub mod interact;
pub mod chart_export;
pub mod markdown;

// ── Utilities ──
pub mod color;
pub mod date;
pub mod schema;

// ── Async & Realtime ──
pub mod query;
pub mod websocket;

// ── Layout & Rendering ──
pub mod toast;
pub mod portal;
pub mod head;
pub mod theme;

// ── Graphics ──
pub mod canvas2d;
pub mod webgl;

// ── Communication ──
pub mod worker;
pub mod sse;
pub mod graphql;
pub mod clipboard;

// ── Security ──
pub mod auth;
pub mod crypto;
pub mod sanitize;

// ── Data Formats ──
pub mod csv;
pub mod search;
pub mod highlight;

// ── Navigation & UX ──
pub mod scroll;
pub mod transition;
pub mod upload;
pub mod pwa;

// ── Browser APIs ──
pub mod indexeddb;
pub mod streams;
pub mod broadcast;
pub mod abort;
pub mod intersection;
pub mod resize;
pub mod geo;
pub mod notification;
pub mod gamepad;
pub mod webrtc;
pub mod speech;
pub mod lock;

// ── Data Processing ──
pub mod pdf;
pub mod zip;
pub mod qrcode;
pub mod image;
pub mod diff;
pub mod xml;
pub mod json_patch;

// ── UI Patterns ──
pub mod command_palette;
pub mod breadcrumb;
pub mod accordion;
pub mod tabs;
pub mod carousel;
pub mod skeleton;
pub mod mask;
pub mod tree;
pub mod floating;

// ── Infrastructure ──
pub mod observable;
pub mod fsm;
pub mod template;
pub mod error_track;
pub mod feature_flag;
pub mod analytics;
pub mod rate_limit;
pub mod immutable;
pub mod pipe;
pub mod test_utils;

// ── 3D Engine ──
pub mod scene3d;
pub mod mesh;
pub mod material;
pub mod camera3d;
pub mod physics3d;
pub mod lighting;
pub mod particle;
pub mod terrain;

// ── Audio ──
pub mod audio_engine;
pub mod synthesizer;
pub mod audio_fx;
pub mod waveform;
pub mod midi;
pub mod equalizer;
pub mod spatial_audio;
pub mod audio_graph;
pub mod audio_buffer;
pub mod oscillator;
pub mod audio_filter;
pub mod envelope;
pub mod delay_line;
pub mod audio_analysis;

// ── Video & Media ──
pub mod video_player;
pub mod hls_stream;
pub mod subtitle;
pub mod media_recorder;
pub mod webcodecs;
pub mod screen_capture;
pub mod media_session;
pub mod playlist;

// ── ML & AI ──
pub mod tensor;
pub mod model_loader;
pub mod vision_pipeline;
pub mod pose_detect;
pub mod face_detect;
pub mod segmentation;
pub mod nlp_token;
pub mod embedding;

// ── Maps & Advanced Viz ──
pub mod map_engine;
pub mod tile_renderer;
pub mod geo_projection;
pub mod marker_cluster;
pub mod scale;
pub mod axis_render;
pub mod force_layout;
pub mod voronoi;

// ── Rich Text & Editors ──
pub mod rich_editor;
pub mod doc_model;
pub mod text_selection;
pub mod spell_check;
pub mod syntax_editor;
pub mod autocomplete;
pub mod diff_editor;
pub mod code_format;

// ── Diagramming ──
pub mod graph_layout;
pub mod diagram;
pub mod flowchart;
pub mod network_viz;
pub mod gantt;
pub mod org_chart;
pub mod mindmap;
pub mod sankey;

// ── File Formats ──
pub mod xlsx;
pub mod docx_gen;
pub mod font_loader;
pub mod opentype;

// ── Advanced Auth & P2P ──
pub mod webauthn;
pub mod oauth2_pkce;
pub mod jwt_codec;
pub mod p2p_mesh;

// ── Payments & Commerce ──
pub mod payment;
pub mod invoice;
pub mod cart;
pub mod currency;
pub mod tax;
pub mod subscription;
pub mod coupon;
pub mod checkout;

// ── Typography & Text ──
pub mod text_shaper;
pub mod bidi;
pub mod hyphenation;
pub mod line_break;
pub mod font_metrics;
pub mod text_wrap;
pub mod ligature;
pub mod unicode_normalize;

// ── Advanced Charts ──
pub mod polar_chart;
pub mod radar_chart;
pub mod candlestick;
pub mod gauge;
pub mod funnel;
pub mod scatter3d;
pub mod parallel_coords;
pub mod graph_chart;

// ── Testing Framework ──
pub mod dom_test;
pub mod visual_regression;
pub mod perf_bench;
pub mod snapshot_test;
pub mod mock_http;
pub mod test_runner;
pub mod coverage;
pub mod assertion;
pub mod property_test;
pub mod mock_builder;
pub mod fixture;
pub mod coverage_tracker;
pub mod benchmark;
pub mod fault_inject;
pub mod test_data_gen;

// ── Build Tools ──
pub mod bundler;
pub mod minifier;
pub mod transpiler;
pub mod source_map;
pub mod hot_reload;
pub mod tree_shake;
pub mod css_modules;
pub mod asset_pipeline;

// ── CSS Layout Engine ──
pub mod css_grid;
pub mod flexbox;
pub mod css_var;
pub mod css_transition_engine;
pub mod selector_engine;
pub mod specificity;
pub mod cascade_resolver;
pub mod box_model;

// ── Database & Storage ──
pub mod orm;
pub mod migration;
pub mod query_builder;
pub mod connection_pool;
pub mod schema_validate;
pub mod data_sync;
pub mod offline_store;
pub mod cache_strategy;

// ── Database & Query Patterns ──
pub mod query_planner;
pub mod btree_index;
pub mod hash_index;
pub mod transaction_log;
pub mod join_algorithms;
pub mod aggregation;
pub mod cursor;

// ── Cache & Storage Patterns ──
pub mod kv_store;
pub mod session_store;
pub mod memory_pool;
pub mod object_pool;
pub mod content_cache;
pub mod write_ahead_log;
pub mod buffer_pool;

// ── DevTools & Monitoring ──
pub mod debugger;
pub mod profiler;
pub mod inspector;
pub mod console_logger;
pub mod network_monitor;
pub mod lighthouse;
pub mod heap_snapshot;
pub mod flame_graph;
pub mod call_graph;
pub mod memory_profiler;
pub mod debug_protocol;
pub mod error_tracker;
pub mod profiler_report;
pub mod invariant_checker;

// ── Animation & Motion ──
pub mod spring_anim;
pub mod tween;
pub mod keyframe_anim;
pub mod motion_path;
pub mod lottie;
pub mod parallax;
pub mod scroll_anim;
pub mod morph;

// ── Advanced State & Reactivity ──
pub mod store;
pub mod store_redux;
pub mod signal;
pub mod atom;
pub mod computed;
pub mod effect;
pub mod selector;
pub mod devtools_state;
pub mod persist;
pub mod middleware;
pub mod event_emitter;
pub mod undo_manager;
pub mod finite_state;
pub mod data_binding;
pub mod query_cache;

// ── Calendar & Scheduling ──
pub mod calendar;
pub mod date_picker;
pub mod scheduler;
pub mod recurrence;
pub mod timezone;
pub mod relative_time;
pub mod duration;
pub mod cron_expr;

// ── Data Grid & Tables ──
pub mod data_grid;
pub mod column_config;
pub mod row_selection;
pub mod cell_editor;
pub mod pivot_table;
pub mod data_export;
pub mod filter_engine;
pub mod sort_engine;

// ── Math & Science ──
pub mod linear_algebra;
pub mod statistics;
pub mod interpolation;
pub mod bezier;
pub mod noise;
pub mod random;
pub mod complex_num;
pub mod fraction;

// ── Math & Geometry ──
pub mod matrix_ops;
pub mod polynomial;
pub mod geometry2d;
pub mod geometry3d;
pub mod stats_engine;
pub mod bigint;
pub mod signal_processing;

// ── Advanced Graphics ──
pub mod svg_path;
pub mod shader;
pub mod webgpu;
pub mod sprite;
pub mod tilemap;
pub mod camera2d;
pub mod collision2d;
pub mod physics2d;

// ── Geometry & Physics Simulation ──
pub mod rigid_body;
pub mod particle_system;
pub mod spring_physics;
pub mod spatial_hash;
pub mod ray_cast;
pub mod curve_math;

// ── Communication & Social ──
pub mod email_template;
pub mod sms;
pub mod notification_center;
pub mod presence;
pub mod activity_feed;
pub mod comment;
pub mod reaction;
pub mod share;

// ── Compression & Encoding ──
pub mod deflate;
pub mod huffman;
pub mod lz77;
pub mod base_encoding;
pub mod url_encode;
pub mod percent_encode;
pub mod hex;
pub mod ascii85;

// ── Platform ──
pub mod storage;
pub mod energy;

// ── Networking & HTTP ──
pub mod http_client;
pub mod cors;
pub mod proxy;
pub mod cookie_jar;
pub mod dns_resolve;
pub mod http2;
pub mod grpc_web;
pub mod sse_client;

// ── Form Validation & Utilities ──
pub mod validator;
pub mod json_schema_validate;
pub mod input_mask;
pub mod form_wizard;
pub mod field_array;
pub mod debounce;
pub mod throttle;
pub mod memoize;

// ── Accessibility ──
pub mod aria;
pub mod focus_trap;
pub mod screen_reader;
pub mod live_region;
pub mod skip_link;
pub mod color_contrast;
pub mod reduced_motion;
pub mod keyboard_nav;

// ── Canvas & Geometry ──
pub mod svg_filter;
pub mod svg_animate;
pub mod canvas_text;
pub mod canvas_composite;
pub mod path_boolean;
pub mod polygon_clip;
pub mod convex_hull;
pub mod triangulation;

// ── Pattern & String Processing ──
pub mod regex_engine;
pub mod glob_match;
pub mod string_distance;
pub mod fuzzy_match;
pub mod template_literal;
pub mod base64_codec;
pub mod crc_hash;
pub mod murmur_hash;

// ── Concurrency & Scheduling ──
pub mod task_queue;
pub mod priority_queue;
pub mod event_loop;
pub mod scheduler_coop;
pub mod channel;
pub mod semaphore;
pub mod backpressure;
pub mod circuit_breaker;

// ── UI Components ──
pub mod modal;
pub mod dropdown;
pub mod tooltip;
pub mod popover;
pub mod stepper;
pub mod avatar;
pub mod badge;
pub mod chip;

// ── Internationalization & Formatting ──
pub mod icu_message;
pub mod plural_rules;
pub mod number_format;
pub mod date_format;
pub mod list_format;
pub mod collation;
pub mod segmenter;
pub mod bidi_algorithm;

// ── Data Visualization ──
pub mod treemap;
pub mod heatmap;
pub mod histogram;
pub mod box_plot;
pub mod waterfall;
pub mod sparkline;
pub mod bullet_chart;
pub mod timeline;

// ── Design System ──
pub mod color_palette;
pub mod spacing;
pub mod breakpoint;
pub mod design_token;
pub mod responsive;
pub mod grid_system;
pub mod container_query;
pub mod media_query;

// ── Logging & Observability ──
pub mod structured_log;
pub mod metrics;
pub mod distributed_trace;
pub mod health_check;
pub mod sli_slo;
pub mod alerting;
pub mod audit_log;
pub mod feature_metric;

// ── Serialization & Encoding ──
pub mod protobuf;
pub mod msgpack;
pub mod cbor;
pub mod toml_parse;
pub mod yaml_parse;
pub mod bencode;
pub mod asn1;
pub mod bson;

// ── ECS & Game Patterns ──
pub mod ecs;
pub mod behavior_tree;
pub mod pathfinding;
pub mod quadtree;
pub mod octree;
pub mod nav_mesh;
pub mod steering;
pub mod game_loop;

// ── Cryptography & Security ──
pub mod sha256;
pub mod sha512;
pub mod aes;
pub mod chacha20;
pub mod pbkdf2;
pub mod csp;
pub mod xss_protect;
pub mod csrf;

// ── Cryptographic Protocols & Primitives ──
pub mod hmac_engine;
pub mod merkle_tree;
pub mod bloom_crypt;
pub mod secret_sharing;
pub mod commitment;
pub mod oblivious_transfer;
pub mod zero_knowledge;
pub mod threshold_crypto;

// ── Security & Auth ──
pub mod rate_limiter;
pub mod csrf_protect;
pub mod cors_handler;
pub mod password_hash;
pub mod jwt_lite;
pub mod permission_engine;
pub mod input_sanitizer;

// ── CLI & Terminal ──
pub mod ansi_color;
pub mod terminal;
pub mod progress_bar;
pub mod spinner;
pub mod prompt;
pub mod table_render;
pub mod arg_parse;
pub mod term_emulator;

// ── File Formats & Parsing ──
pub mod csv_parse;
pub mod ini_parse;
pub mod xml_parse;
pub mod markdown_parse;
pub mod cron_parse;
pub mod semver_parse;
pub mod mime_type;

// ── Data Structures ──
pub mod bloom_filter;
pub mod skip_list;
pub mod trie;
pub mod lru_cache;
pub mod rope;
pub mod interval_tree;
pub mod disjoint_set;
pub mod ring_buffer;

// ── Advanced Data Structures ──
pub mod red_black_tree;
pub mod avl_tree;
pub mod segment_tree;
pub mod fenwick_tree;
pub mod splay_tree;
pub mod treap;
pub mod suffix_array;
pub mod wavelet_tree;

// ── Image Processing ──
pub mod pixel_buffer;
pub mod image_filter;
pub mod color_space;
pub mod histogram_image;
pub mod morphology;
pub mod connected_component;
pub mod image_blend;
pub mod dither;

// ── Network Protocols ──
pub mod dns_packet;
pub mod mqtt;
pub mod websocket_frame;
pub mod http_parse;
pub mod smtp;
pub mod redis_protocol;
pub mod json_rpc;
pub mod uri_parse;

// ── Template & Document Generation ──
pub mod mustache;
pub mod handlebars_engine;
pub mod liquid_template;
pub mod text_diff_engine;
pub mod pdf_layout;
pub mod email_builder;
pub mod svg_builder;
pub mod html_builder;

// ── Workflow & Automation ──
pub mod workflow;
pub mod rule_engine;
pub mod state_machine;
pub mod saga;
pub mod cqrs;
pub mod scheduler_job;
pub mod pipeline;
pub mod webhook;

// ── Diff, Patch & Merge ──
pub mod myers_diff;
pub mod unified_diff;
pub mod json_diff;
pub mod three_way_merge;
pub mod operational_transform;
pub mod crdt_text;
pub mod change_set;
pub mod semantic_diff;

// ── Queue & Messaging ──
pub mod message_queue;
pub mod pub_sub;
pub mod event_bus;
pub mod command_bus;
pub mod job_queue;
pub mod stream_buffer;
pub mod message_broker;
pub mod outbox;

// ── Configuration & Feature Flags ──
pub mod config_loader;
pub mod env_parse;
pub mod secret_manager;
pub mod config_watch;
pub mod ab_test;
pub mod config_merge;

// ── Search & Indexing ──
pub mod inverted_index;
pub mod bm25;
pub mod spell_checker;
pub mod autocomplete_engine;
pub mod search_query_parse;
pub mod text_tokenizer;
pub mod search_highlight;
pub mod faceted_search;

// ── Process & Runtime Patterns ──
pub mod worker_pool;
pub mod actor_model;
pub mod future_combinator;
pub mod coroutine;
pub mod supervisor;
pub mod scheduler_round_robin;
pub mod dependency_injector;
pub mod plugin_system;

// ── API & GraphQL ──
pub mod graphql_parse;
pub mod graphql_execute;
pub mod openapi_spec;
pub mod api_versioning;
pub mod rest_resource;
pub mod api_rate_limit;
pub mod webhook_dispatch;
pub mod content_negotiation;

// ── ML & Numerical Methods ──
pub mod knn;
pub mod decision_tree;
pub mod naive_bayes;
pub mod optimization;
pub mod kmeans;
pub mod neural_net;
pub mod dimensionality;

// ── Network & Transport ──
pub mod tcp_state;
pub mod ip_addr_util;
pub mod packet_codec;
pub mod nat_traversal;
pub mod load_balancer;
pub mod service_registry;
pub mod circuit_breaker_adv;
pub mod protocol_mux;

// ── Scheduling & Time Management ──
pub mod timer_wheel;
pub mod retry_policy;
pub mod deadline;
pub mod rate_controller;
pub mod temporal_index;
pub mod time_series;
pub mod task_dag;
pub mod recurring_task;

// ── Compiler & Language Tools ──
pub mod lexer;
pub mod pratt_parser;
pub mod type_checker;
pub mod bytecode_vm;
pub mod regex_compile;
pub mod cfg_grammar;
pub mod ir_builder;
pub mod symbol_table;

// ── Encoding & Advanced Formats ──
pub mod varint;
pub mod bitstream;
pub mod rle_encoding;
pub mod escaped_string;
pub mod hex_dump;
pub mod unicode_util;

// ── Graph Algorithms & Distributed Patterns ──
pub mod graph;
pub mod shortest_path;
pub mod min_spanning;
pub mod network_flow;
pub mod consensus;
pub mod consistent_hash;
pub mod vector_clock;
pub mod gossip_protocol;

// ── Distributed Systems ──
pub mod raft_log;
pub mod paxos;
pub mod crdt_counter;
pub mod crdt_set;
pub mod crdt_register;
pub mod lamport_clock;
pub mod membership;
pub mod sharding;

// ── Functional Programming Patterns ──
pub mod monad;
pub mod lens;
pub mod pattern_match;
pub mod lazy_eval;
pub mod persistent_ds;
pub mod functor;
pub mod pipe_compose;
pub mod transducer;

// ── OS & System Patterns ──
pub mod virtual_fs;
pub mod process_tree;
pub mod memory_allocator;
pub mod page_replacement;
pub mod disk_scheduler;
pub mod ipc_channel;
pub mod file_system_journal;
pub mod resource_manager;

// ── Domain-Driven Design & Modeling ──
pub mod entity;
pub mod value_object;
pub mod aggregate;
pub mod repository_pattern;
pub mod specification;
pub mod domain_event;
pub mod bounded_context;
pub mod money;

// ── Content & Media Processing ──
pub mod markdown_ext;
pub mod syntax_highlight;
pub mod content_pipeline;
pub mod feed_generator;
pub mod sitemap_builder;
pub mod slug_generator;
pub mod image_metadata;
pub mod media_playlist;

// ── Monitoring & Operations ──
pub mod anomaly_detect;
pub mod capacity_planner;
pub mod incident_tracker;
pub mod canary_deploy;
pub mod chaos_engine;
pub mod log_aggregator;
pub mod trace_context;
pub mod uptime_monitor;

// ── Data Pipeline & ETL ──
pub mod data_pipeline_etl;
pub mod data_validator;
pub mod data_transform;
pub mod dedup_engine;
pub mod data_sampler;
pub mod schema_evolution;
pub mod data_quality;
pub mod batch_processor;

// ── WebAssembly & Runtime Concepts ──
pub mod wasm_module;
pub mod wasm_interp;
pub mod wasm_memory;
pub mod sandbox;
pub mod jit_concept;
pub mod gc_collector;
pub mod module_loader;
pub mod capability;

// ── UI Framework Core ──
pub mod vdom_diff;
pub mod component_lifecycle;
pub mod render_tree;
pub mod style_resolver;
pub mod event_delegation;
pub mod reconciler;
pub mod hydration;

// ── NLP & Text Analysis ──
pub mod tokenize_nlp;
pub mod stemmer;
pub mod stop_words;
pub mod tfidf;
pub mod sentiment;
pub mod text_classify;
pub mod ner_tagger;
pub mod text_summarize;

// ── Blockchain & Ledger Concepts ──
pub mod blockchain;
pub mod ledger;
pub mod smart_contract;
pub mod token_ledger;
pub mod nft_registry;
pub mod tx_pool;
pub mod state_trie;
pub mod consensus_pow;

// ── Metrics & Telemetry ──
pub mod prometheus_fmt;
pub mod histogram_metrics;
pub mod meter;
pub mod reservoir;
pub mod metric_registry;
pub mod span_metrics;
pub mod resource_metrics;
pub mod metric_export;

// ── Compiler Backend & Code Generation ──
pub mod register_alloc;

// ── CLI & Developer Tools ──
pub mod repl;
pub mod task_runner;
pub mod code_formatter;
pub mod linter_engine;
pub mod changelog;
pub mod dep_graph;
pub mod doc_generator;
pub mod scaffold;
pub mod instruction_select;
pub mod codegen_emit;
pub mod optimizer_pass;
pub mod dominance;
pub mod liveness;
pub mod peephole;
pub mod linker;

// ── Advanced Networking ──
pub mod http_router;
pub mod http_middleware;
pub mod sse_protocol;
pub mod http_cache;
pub mod oauth_flow;
pub mod api_pagination;
pub mod webhook_verify;
pub mod request_dedup;

// ── Advanced Visualization & Charts ──
pub mod sankey_diagram;
pub mod chord_diagram;
pub mod gauge_chart;
pub mod funnel_chart;
pub mod network_graph;
pub mod calendar_heatmap;
pub mod density_plot;

// ── Error Handling & Resilience ──
pub mod error_chain;
pub mod fallback;
pub mod bulkhead;
pub mod hedge_request;
pub mod timeout_manager;
pub mod error_budget;
pub mod graceful_degradation;
pub mod compensation;

// ── IoT & Embedded Patterns ──
pub mod sensor_fusion;
pub mod device_registry;
pub mod telemetry_ingest;
pub mod ota_update;
pub mod edge_compute;
pub mod gpio_sim;
pub mod i2c_sim;
pub mod power_mgmt;

// ── Type System & Generic Abstractions ──
pub mod type_registry;
pub mod schema_type;
pub mod tagged_union;
pub mod newtype;
pub mod result_ext;
pub mod builder_derive;
pub mod phantom_types;
pub mod generic_collection;

// ── Classic Design Patterns ──
pub mod strategy_pattern;
pub mod chain_of_responsibility;
pub mod mediator;
pub mod interpreter;
pub mod flyweight;
pub mod proxy_pattern;
pub mod memento;
pub mod object_mapper;

// ── Simulation & Modeling ──
pub mod cellular_automaton;
pub mod monte_carlo;
pub mod markov_chain;
pub mod queuing_sim;
pub mod agent_sim;
pub mod petri_net;
pub mod fluid_sim;
pub mod epidemic_sim;

// ── Event Sourcing & CQRS Extended ──
pub mod event_store;
pub mod projection;
pub mod snapshot_store;
pub mod process_manager;
pub mod read_model;
pub mod event_migration;
pub mod command_handler;
pub mod event_subscription;

// ── Access Control & Policy ──
pub mod abac;
pub mod policy_engine;
pub mod tenant_isolation;
pub mod data_masking;
pub mod access_log;
pub mod ip_filter;
pub mod api_key_mgr;
pub mod session_mgr;

// ── Protocol Buffers & RPC ──
pub mod proto_schema;
pub mod proto_encode;
pub mod proto_decode;
pub mod rpc_framework;
pub mod rpc_middleware;
pub mod rpc_discovery;
pub mod proto_reflect;
pub mod rpc_testing;

// ── RPC & Wire Protocol ──
pub mod binary_rpc;
pub mod rpc_codec;
pub mod rpc_retry;
pub mod wire_format;
pub mod rpc_stream;
pub mod rpc_batch;
pub mod service_stub;
pub mod rpc_intercept;

// ── Numerical Computing ──
pub mod sparse_matrix;
pub mod ode_solver;
pub mod root_finding;
pub mod quadrature;
pub mod finite_diff;
pub mod random_dist;

// ── File System & Storage ──
pub mod lsm_tree;
pub mod sstable;
pub mod wal_manager;
pub mod extent_alloc;
pub mod block_device;
pub mod file_index;
pub mod versioned_store;
pub mod compaction;

// ── Workflow & Automation 2 ──
pub mod approval_workflow;
pub mod notification_engine;
pub mod form_engine;
pub mod report_builder;
pub mod batch_scheduler;
pub mod integration_hub;
pub mod template_processor;

// ── Auth & Security Web ──
pub mod oauth2_flow;
pub mod oidc;
pub mod csrf_protection;
pub mod security_headers;

// ── API & Transport ──
pub mod http2_frame;
pub mod graphql_schema;
pub mod jsonrpc;
pub mod multipart;
pub mod etag;

// ── i18n & Content ──
pub mod l10n;
pub mod pdf_gen;
pub mod image_ops;
pub mod qr_code;
pub mod sitemap;
pub mod feed_gen;

// ── Modern Frontend Patterns ──
pub mod pwa_manifest;
pub mod service_worker;
pub mod css_in_rs;
pub mod seo;
pub mod ab_testing;

// ── Data & Validation ──
pub mod json_pointer;
pub mod csv_engine;
pub mod semver;

// ── Infra & DevOps ──
pub mod feature_gate;
pub mod blue_green;
pub mod chaos_eng;

// ── Realtime & Messaging ──
pub mod webrtc_signal;
pub mod live_cursor;
pub mod collab_edit;
pub mod push_notify;
pub mod long_poll;
pub mod channel_mux;

// ── Testing & DX ──
pub mod http_mock;
pub mod contract_test;
pub mod load_profile;
pub mod fixture_factory;
pub mod error_page;

// ── Production Ops ──
pub mod graceful_shutdown;
pub mod request_id;
pub mod response_cache;
pub mod compression_negotiate;
pub mod request_coalesce;

// ── Observability & Monitoring ──
pub mod sla_tracker;
pub mod cost_tracker;
pub mod audit_trail;
pub mod benchmark_harness;

// ── Data Pipeline ──
pub mod data_pipeline;
pub mod stream_window;
pub mod change_data_capture;
pub mod dead_letter;
pub mod idempotency;
pub mod saga_orchestrator;
pub mod schema_registry;

// ── API Gateway Patterns ──
pub mod api_gateway;
pub mod service_mesh;
pub mod request_transform;
pub mod response_transform;
pub mod traffic_mirror;
pub mod jwt_gateway;

// ── Developer Tooling ──
pub mod code_gen;
pub mod dependency_check;
pub mod migration_runner;
pub mod env_validator;
pub mod openapi_client;
pub mod debug_toolbar;

// ── State & Caching ──
pub mod cache_aside;
pub mod distributed_lock;
pub mod pagination;
pub mod search_index;
pub mod content_store;

// ── Auth & Identity ──
pub mod rbac;
pub mod mfa;
pub mod social_auth;
pub mod token_bucket_auth;
pub mod permission_cache;
pub mod invite_system;
pub mod user_profile;
pub mod org_hierarchy;

// ── Media & Encoding ──
pub mod media_type;
pub mod content_disposition;
pub mod data_uri;
pub mod canvas_2d;
pub mod sprite_sheet;

// ── Game Engine ECS ──
pub mod entity_store;
pub mod component_registry;
pub mod system_scheduler;
pub mod archetype_storage;
pub mod query_engine_ecs;
pub mod world_state;
pub mod entity_commands;
pub mod change_detection_ecs;

// ── Game Input & Mapping ──
pub mod input_manager;
pub mod gamepad_input;
pub mod touch_gesture_game;
pub mod key_binding_game;
pub mod action_mapping;
pub mod input_replay;
pub mod cursor_lock;
pub mod haptic_feedback;

// ── Game Loop & Scene Management ──
pub mod game_tick_loop;
pub mod scene_graph_3d;
pub mod scene_manager;
pub mod camera_system;
pub mod viewport_manager;
pub mod render_queue;
pub mod draw_call_batcher;
pub mod frame_graph;

// ── Game AI & Navigation ──
pub mod navmesh;
pub mod pathfind_astar;
pub mod flow_field_path;
pub mod steering_behavior;
pub mod behavior_tree_ai;
pub mod utility_ai;
pub mod flocking_sim;
pub mod formation;

// ── 2D Physics Engine ──
pub mod rigid_body_2d;
pub mod collision_2d;
pub mod broadphase_2d;
pub mod narrowphase_2d;
pub mod joint_2d;
pub mod contact_solver_2d;
pub mod spatial_grid_2d;
pub mod physics_world_2d;

// ── 3D Physics Engine ──
pub mod rigid_body_3d;
pub mod collision_3d;
pub mod broadphase_3d;
pub mod gjk_epa;
pub mod contact_manifold;
pub mod constraint_solver_3d;
pub mod ragdoll_physics;
pub mod vehicle_physics;

// ── 3D Rendering Fundamentals ──
pub mod mesh_builder;
pub mod vertex_format;
pub mod index_buffer;
pub mod transform_hierarchy;
pub mod bounding_volume;
pub mod frustum_cull;
pub mod lod_system;
pub mod instanced_draw;
pub mod gpu_particles;
pub mod particle_emitter;
pub mod particle_affector;
pub mod ribbon_trail;
pub mod beam_renderer;
pub mod decal_projector;
pub mod distortion_fx;
pub mod volumetric_fog;

// ── PBR Materials & Shaders ──
pub mod pbr_material;
pub mod shader_graph;
pub mod texture_sampler;
pub mod normal_mapping;
pub mod parallax_mapping;
pub mod subsurface_scatter;
pub mod clearcoat;
pub mod anisotropic_material;

// ── Lighting Engine ──
pub mod directional_light;
pub mod point_light;
pub mod spot_light;
pub mod area_light;
pub mod shadow_cascade;
pub mod shadow_atlas;
pub mod light_probe;
pub mod irradiance_volume;

// ── Post-Processing Effects ──
pub mod bloom_fx;
pub mod motion_blur;
pub mod depth_of_field;
pub mod chromatic_aberration;
pub mod tone_mapper;
pub mod color_grading_render;
pub mod ssao;
pub mod screen_space_reflect;

// ── Rendering Quality & Anti-Aliasing ──
pub mod fxaa;
pub mod taa;
pub mod msaa_resolve;
pub mod sharpen_filter;
pub mod upscaler;
pub mod mipmap_gen;
pub mod aniso_filter;
pub mod dithering;

// ── Terrain & Environment ──
pub mod heightmap_terrain;
pub mod terrain_splat;
pub mod grass_renderer;
pub mod tree_billboard;
pub mod water_surface;
pub mod sky_atmosphere;
pub mod cloud_renderer;
pub mod weather_system;

// ── Text & UI Rendering ──
pub mod sdf_font;
pub mod msdf_text;
pub mod text_layout_engine;
pub mod glyph_cache;
pub mod ui_renderer;
pub mod nine_slice;
pub mod signed_distance_2d;
pub mod vector_rasterizer;

// ── Ray Tracing ──
pub mod ray_intersect;
pub mod bvh_builder;
pub mod path_tracer;
pub mod photon_map;
pub mod denoiser;
pub mod importance_sample;
pub mod monte_carlo_render;
pub mod brdf_model;

// ── Audio Engine ──
pub mod audio_graph_engine;
pub mod audio_mixer;
pub mod spatial_audio_3d;
pub mod hrtf;
pub mod reverb_engine;
pub mod audio_bus;
pub mod crossfade_audio;
pub mod audio_pool;

// ── Audio Synthesis ──
pub mod oscillator_synth;
pub mod envelope_adsr;
pub mod audio_filter_dsp;
pub mod wavetable;
pub mod fm_synthesis;
pub mod granular_synth;
pub mod noise_gen_audio;
pub mod vocoder;

// ── Music Systems ──
pub mod midi_parser;
pub mod sequencer;
pub mod arpeggiator;
pub mod chord_progression;
pub mod scale_quantize;
pub mod tempo_sync;
pub mod music_theory;
pub mod procedural_music;

// ── Procedural Terrain ──
pub mod perlin_noise;
pub mod simplex_noise;
pub mod worley_noise;
pub mod fractal_brownian;
pub mod erosion_sim;
pub mod biome_generator;
pub mod cave_generator;
pub mod river_network;

// ── Procedural Content ──
pub mod dungeon_generator;
pub mod maze_builder;
pub mod room_placer;
pub mod corridor_linker;
pub mod loot_table;
pub mod name_generator;
pub mod texture_synth;
pub mod l_system;

// ── Procedural Geometry ──
pub mod parametric_surface;
pub mod extrude_path;
pub mod lathe_mesh;
pub mod subdivision_surface;
pub mod catmull_clark;
pub mod loop_subdivision;
pub mod mesh_simplify;
pub mod mesh_boolean;

// ── World Streaming & Assets ──
pub mod chunk_manager;
pub mod infinite_terrain;
pub mod level_streaming;
pub mod asset_loader;
pub mod resource_cache_game;
pub mod hot_reload_assets;
pub mod prefab_system;
pub mod object_pool_game;

// ── Game Systems ──
pub mod inventory_system;
pub mod crafting_system;
pub mod dialogue_tree;
pub mod quest_tracker;
pub mod achievement_system;
pub mod save_game;
pub mod replay_recorder;
pub mod leaderboard;

// ── Fluid Dynamics ──
pub mod sph_fluid;
pub mod eulerian_fluid;
pub mod navier_stokes;
pub mod pressure_solver;
pub mod velocity_field;
pub mod vortex_particle;
pub mod fluid_surface;
pub mod fluid_coupling;

// ── Soft Body Simulation ──
pub mod mass_spring;
pub mod cloth_sim;
pub mod deformable_mesh;
pub mod shape_matching;
pub mod position_based_dynamics;
pub mod volume_preserve;
pub mod tear_sim;
pub mod rope_physics;

// ── N-Body & Orbital Mechanics ──
pub mod nbody_sim;
pub mod barnes_hut;
pub mod orbital_mechanics;
pub mod gravity_field;
pub mod celestial_render;
pub mod star_catalog;
pub mod galaxy_gen;
pub mod trajectory_planner;

// ── Cellular Automata & Swarm ──
pub mod cellular_automaton_2d;
pub mod game_of_life;
pub mod langton_ant;
pub mod reaction_diffusion;
pub mod boids_advanced;
pub mod ant_colony;
pub mod swarm_intelligence;
pub mod crowd_sim;

// ── Optimization Algorithms ──
pub mod genetic_algorithm;
pub mod particle_swarm;
pub mod simulated_annealing;
pub mod hill_climbing;
pub mod evolutionary_strategy;
pub mod multi_objective;
pub mod constraint_satisfy;
pub mod bayesian_opt;

// ── Signal Processing ──
pub mod fft_engine;
pub mod stft;
pub mod mel_spectrogram;
pub mod bandpass_filter;
pub mod iir_filter;
pub mod fir_filter;
pub mod window_function;
pub mod convolution_engine;

// ── Control Systems ──
pub mod pid_controller;
pub mod kalman_filter;
pub mod extended_kalman;
pub mod complementary_filter;
pub mod model_predictive;
pub mod state_observer;
pub mod feedforward;
pub mod adaptive_control;

// ── Numerical Linear Algebra ──
pub mod sparse_matrix_solver;
pub mod conjugate_gradient;
pub mod lu_decompose;
pub mod qr_decompose;
pub mod svd_decompose;
pub mod eigenvalue_solver;
pub mod least_squares;
pub mod polynomial_fit;

// ── Peer-to-Peer ──
pub mod peer_discovery;
pub mod relay_server;
pub mod hole_punch;
pub mod dht_store;
pub mod peer_auth;
pub mod p2p_relay;
pub mod mesh_topology;
pub mod peer_score;

// ── Lobby & Matchmaking ──
pub mod lobby_system;
pub mod matchmaker;
pub mod matchmaker_elo;
pub mod session_relay;
pub mod room_manager_net;
pub mod team_balance;
pub mod queue_match;
pub mod region_select;

// ── Realtime Sync ──
pub mod state_sync;
pub mod clock_sync;
pub mod interest_mgmt;
pub mod entity_replication;
pub mod conflict_resolve_net;
pub mod eventual_consist;
pub mod authority_model;
pub mod bandwidth_throttle;

// ── Netcode ──
pub mod netcode_predict;
pub mod netcode_snapshot;
pub mod netcode_rollback;
pub mod lag_compensate;
pub mod delta_compress_net;
pub mod tick_sync;
pub mod jitter_buffer_net;
pub mod packet_loss_sim;

// ── Chat & Social ──
pub mod chat_room;
pub mod chat_history;
pub mod online_status;
pub mod friend_list;
pub mod guild_system;
pub mod voice_lobby;
pub mod emote_system;
pub mod message_dispatch;

// ── Transport Layer ──
pub mod udp_reliable;
pub mod quic_transport;
pub mod transport_mux;
pub mod congestion_ctrl;
pub mod flow_control_net;
pub mod connection_migrate;
pub mod transport_encrypt;
pub mod keepalive_monitor;

// ── Anti-Cheat ──
pub mod anti_cheat;
pub mod speed_hack_detect;
pub mod aimbot_detect;
pub mod replay_validate;
pub mod server_authority;
pub mod input_validate_net;
pub mod cheat_heuristic;
pub mod ban_system;

// ── Neural Net Layers ──
pub mod dense_layer;
pub mod conv2d_layer;
pub mod pooling_layer;
pub mod batch_norm;
pub mod dropout_layer;
pub mod activation_fn;
pub mod embedding_layer_nn;
pub mod attention_layer;

// ── Training & Optimization ──
pub mod gradient_descent;
pub mod adam_optimizer;
pub mod lr_scheduler;
pub mod loss_function;
pub mod backprop_engine;
pub mod weight_init;
pub mod regularization;
pub mod early_stopping;

// ── Transformer Architecture ──
pub mod self_attention;
pub mod multi_head_attn;
pub mod positional_encode;
pub mod feed_forward_block;
pub mod layer_norm_transformer;
pub mod transformer_encoder;
pub mod transformer_decoder;
pub mod beam_search;

// ── Inference Engine ──
pub mod model_runtime;
pub mod tensor_graph;
pub mod quantize_engine;
pub mod pruning;
pub mod knowledge_distill;
pub mod onnx_parse;
pub mod model_cache_infer;
pub mod batch_inference;

// ── Computer Vision ──
pub mod feature_detect_cv;
pub mod edge_detect_cv;
pub mod template_match;
pub mod optical_flow;
pub mod object_track;
pub mod stereo_depth;
pub mod camera_calibrate;
pub mod image_pyramid;

// ── Reinforcement Learning ──
pub mod q_learning;
pub mod policy_gradient;
pub mod replay_buffer_rl;
pub mod environment_sim;
pub mod reward_shaping;
pub mod multi_agent_rl;
pub mod bandit_algo;
pub mod td_learning;

// ── ML Data Pipeline ──
pub mod data_loader_ml;
pub mod augmentation;
pub mod normalization_ml;
pub mod feature_engineer;
pub mod cross_validation;
pub mod confusion_matrix;
pub mod roc_curve;
pub mod precision_recall;

// ── Generative Models ──
pub mod autoencoder;
pub mod variational_ae;
pub mod gan_engine;
pub mod diffusion_model;
pub mod flow_model;
pub mod latent_space;
pub mod style_transfer_nn;
pub mod image_generate;

// ── Kinematics ──
pub mod forward_kinematics;
pub mod inverse_kinematics;
pub mod dh_parameters;
pub mod trajectory_gen;
pub mod motion_profile;
pub mod joint_space;
pub mod workspace_analysis;
pub mod serial_chain;

// ── Motion Planning ──
pub mod rrt_planner;
pub mod prm_planner;
pub mod potential_field;
pub mod a_star_3d;
pub mod velocity_obstacle;
pub mod dynamic_window;
pub mod lattice_planner;
pub mod motion_primitive;

// ── SLAM & Mapping ──
pub mod ekf_slam;
pub mod particle_filter_slam;
pub mod occupancy_grid;
pub mod point_cloud_proc;
pub mod icp_match;
pub mod loop_closure;
pub mod pose_graph;
pub mod visual_odometry;

// ── Perception & Sensors ──
pub mod lidar_process;
pub mod radar_process;
pub mod imu_fusion;
pub mod depth_camera;
pub mod sonar_range;
pub mod encoder_sensor;
pub mod gps_filter;
pub mod sensor_model;

// ── Control & Actuation ──
pub mod motor_control;
pub mod servo_driver;
pub mod pwm_control;
pub mod stepper_control;
pub mod force_torque;
pub mod impedance_control;
pub mod admittance_control;
pub mod hybrid_force_pos;

// ── Autonomous Navigation ──
pub mod global_planner;
pub mod local_planner;
pub mod obstacle_avoid;
pub mod waypoint_follow;
pub mod lane_detect;
pub mod traffic_sign;
pub mod behavior_plan;
pub mod mission_planner;

// ── Manipulation ──
pub mod grasp_planner;
pub mod pick_place;
pub mod push_manip;
pub mod compliant_grasp;
pub mod bin_picking;
pub mod assembly_plan;
pub mod tool_use_robot;
pub mod deformable_manip;

// ── Multi-Robot Systems ──
pub mod formation_control;
pub mod task_allocation;
pub mod swarm_robot;
pub mod comm_robot;
pub mod consensus_robot;
pub mod coverage_plan;
pub mod auction_allocate;
pub mod fleet_manage;

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
pub mod payment_flow;
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
pub mod ntt_engine;
pub mod ntru_encrypt;

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
