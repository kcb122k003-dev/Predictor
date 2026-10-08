-- Schema of version 1 databases (commit f47a6bf), used by the migration test.
CREATE TABLE analysis_artifact (
	id INTEGER NOT NULL, 
	run_id INTEGER NOT NULL, 
	"key" VARCHAR(80) NOT NULL, 
	data JSON NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_artifact_key UNIQUE (run_id, "key"), 
	FOREIGN KEY(run_id) REFERENCES analysis_run (id) ON DELETE CASCADE
);
CREATE TABLE analysis_run (
	id INTEGER NOT NULL, 
	course_id INTEGER NOT NULL, 
	status VARCHAR(20) NOT NULL, 
	progress FLOAT NOT NULL, 
	message TEXT NOT NULL, 
	started_at DATETIME NOT NULL, 
	finished_at DATETIME, 
	config JSON NOT NULL, 
	data_fingerprint VARCHAR(64) NOT NULL, 
	summary JSON NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(course_id) REFERENCES course (id) ON DELETE CASCADE
);
CREATE TABLE backtest_fold (
	id INTEGER NOT NULL, 
	run_id INTEGER NOT NULL, 
	layer VARCHAR(20) NOT NULL, 
	model_name VARCHAR(60) NOT NULL, 
	target_exam_id INTEGER, 
	target_index INTEGER NOT NULL, 
	target_label VARCHAR(120) NOT NULL, 
	n_train_exams INTEGER NOT NULL, 
	metrics JSON NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(run_id) REFERENCES analysis_run (id) ON DELETE CASCADE
);
CREATE TABLE course (
	id INTEGER NOT NULL, 
	name VARCHAR(200) NOT NULL, 
	code VARCHAR(60) NOT NULL, 
	description TEXT NOT NULL, 
	created_at DATETIME NOT NULL, 
	settings JSON NOT NULL, 
	PRIMARY KEY (id)
);
CREATE TABLE course_topic (
	id INTEGER NOT NULL, 
	course_id INTEGER NOT NULL, 
	syllabus_version_id INTEGER, 
	parent_id INTEGER, 
	depth INTEGER NOT NULL, 
	number VARCHAR(40) NOT NULL, 
	title VARCHAR(500) NOT NULL, 
	description TEXT NOT NULL, 
	concepts JSON NOT NULL, 
	objectives JSON NOT NULL, 
	hours FLOAT, 
	marks_weight FLOAT, 
	kinds JSON NOT NULL, 
	aliases JSON NOT NULL, 
	source_refs JSON NOT NULL, 
	order_no INTEGER NOT NULL, 
	excluded BOOLEAN NOT NULL, 
	user_edited BOOLEAN NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(course_id) REFERENCES course (id) ON DELETE CASCADE, 
	FOREIGN KEY(syllabus_version_id) REFERENCES syllabus_version (id) ON DELETE CASCADE, 
	FOREIGN KEY(parent_id) REFERENCES course_topic (id) ON DELETE CASCADE
);
CREATE TABLE document_page (
	id INTEGER NOT NULL, 
	file_id INTEGER NOT NULL, 
	page_no INTEGER NOT NULL, 
	text TEXT NOT NULL, 
	method VARCHAR(20) NOT NULL, 
	ocr_confidence FLOAT, 
	quality_flags JSON NOT NULL, 
	details JSON NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(file_id) REFERENCES source_file (id) ON DELETE CASCADE
);
CREATE TABLE embedding_cache (
	"key" VARCHAR(64) NOT NULL, 
	backend VARCHAR(120) NOT NULL, 
	dim INTEGER NOT NULL, 
	vector BLOB NOT NULL, 
	PRIMARY KEY ("key")
);
CREATE TABLE exam (
	id INTEGER NOT NULL, 
	course_id INTEGER NOT NULL, 
	source_file_id INTEGER, 
	title VARCHAR(400) NOT NULL, 
	subject VARCHAR(400) NOT NULL, 
	year INTEGER, 
	calendar VARCHAR(10) NOT NULL, 
	session VARCHAR(60) NOT NULL, 
	exam_type VARCHAR(60) NOT NULL, 
	exam_date VARCHAR(40) NOT NULL, 
	order_index FLOAT NOT NULL, 
	full_marks FLOAT, 
	pass_marks FLOAT, 
	duration VARCHAR(40) NOT NULL, 
	examiner VARCHAR(200) NOT NULL, 
	instructions JSON NOT NULL, 
	structure JSON NOT NULL, 
	metadata_confidence JSON NOT NULL, 
	include_in_analysis BOOLEAN NOT NULL, 
	exclusion_reason VARCHAR(400) NOT NULL, 
	duplicate_of_id INTEGER, 
	user_edited_fields JSON NOT NULL, 
	created_at DATETIME NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(course_id) REFERENCES course (id) ON DELETE CASCADE, 
	FOREIGN KEY(source_file_id) REFERENCES source_file (id) ON DELETE SET NULL, 
	FOREIGN KEY(duplicate_of_id) REFERENCES exam (id) ON DELETE SET NULL
);
CREATE TABLE exam_question (
	id INTEGER NOT NULL, 
	exam_id INTEGER NOT NULL, 
	section_id INTEGER, 
	parent_id INTEGER, 
	label VARCHAR(20) NOT NULL, 
	path_label VARCHAR(60) NOT NULL, 
	depth INTEGER NOT NULL, 
	text TEXT NOT NULL, 
	raw_text TEXT NOT NULL, 
	normalized_text TEXT NOT NULL, 
	context_text TEXT NOT NULL, 
	marks FLOAT, 
	marks_source VARCHAR(30) NOT NULL, 
	or_group VARCHAR(40), 
	is_optional BOOLEAN NOT NULL, 
	is_leaf BOOLEAN NOT NULL, 
	order_no INTEGER NOT NULL, 
	page_no INTEGER, 
	line_no INTEGER, 
	question_types JSON NOT NULL, 
	type_scores JSON NOT NULL, 
	type_user_edited BOOLEAN NOT NULL, 
	options JSON NOT NULL, 
	equations JSON NOT NULL, 
	quality_flags JSON NOT NULL, 
	parse_confidence FLOAT NOT NULL, 
	needs_review BOOLEAN NOT NULL, 
	user_edited BOOLEAN NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(exam_id) REFERENCES exam (id) ON DELETE CASCADE, 
	FOREIGN KEY(section_id) REFERENCES exam_section (id) ON DELETE SET NULL, 
	FOREIGN KEY(parent_id) REFERENCES exam_question (id) ON DELETE CASCADE
);
CREATE TABLE exam_section (
	id INTEGER NOT NULL, 
	exam_id INTEGER NOT NULL, 
	label VARCHAR(60) NOT NULL, 
	title VARCHAR(300) NOT NULL, 
	instructions TEXT NOT NULL, 
	attempt_count INTEGER, 
	order_no INTEGER NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(exam_id) REFERENCES exam (id) ON DELETE CASCADE
);
CREATE TABLE model_result (
	id INTEGER NOT NULL, 
	run_id INTEGER NOT NULL, 
	layer VARCHAR(20) NOT NULL, 
	model_name VARCHAR(60) NOT NULL, 
	display_name VARCHAR(120) NOT NULL, 
	family VARCHAR(30) NOT NULL, 
	complexity INTEGER NOT NULL, 
	enabled BOOLEAN NOT NULL, 
	gate_reason TEXT NOT NULL, 
	selected BOOLEAN NOT NULL, 
	metrics JSON NOT NULL, 
	metric_se JSON NOT NULL, 
	notes TEXT NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(run_id) REFERENCES analysis_run (id) ON DELETE CASCADE
);
CREATE TABLE predicted_question (
	id INTEGER NOT NULL, 
	run_id INTEGER NOT NULL, 
	topic_id INTEGER, 
	text TEXT NOT NULL, 
	question_type VARCHAR(60) NOT NULL, 
	marks_low FLOAT, 
	marks_high FLOAT, 
	basis VARCHAR(40) NOT NULL, 
	rank INTEGER NOT NULL, 
	evidence_question_ids JSON NOT NULL, 
	grounding JSON NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(run_id) REFERENCES analysis_run (id) ON DELETE CASCADE, 
	FOREIGN KEY(topic_id) REFERENCES course_topic (id) ON DELETE SET NULL
);
CREATE TABLE prediction (
	id INTEGER NOT NULL, 
	run_id INTEGER NOT NULL, 
	layer VARCHAR(20) NOT NULL, 
	topic_id INTEGER, 
	item_key VARCHAR(80) NOT NULL, 
	label VARCHAR(500) NOT NULL, 
	rank INTEGER NOT NULL, 
	score FLOAT NOT NULL, 
	probability FLOAT, 
	prob_low FLOAT, 
	prob_high FLOAT, 
	calibrated BOOLEAN NOT NULL, 
	category VARCHAR(30) NOT NULL, 
	confidence VARCHAR(20) NOT NULL, 
	features JSON NOT NULL, 
	contributions JSON NOT NULL, 
	evidence JSON NOT NULL, 
	why_not JSON NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(run_id) REFERENCES analysis_run (id) ON DELETE CASCADE, 
	FOREIGN KEY(topic_id) REFERENCES course_topic (id) ON DELETE SET NULL
);
CREATE TABLE question_topic_mapping (
	id INTEGER NOT NULL, 
	question_id INTEGER NOT NULL, 
	topic_id INTEGER, 
	rank INTEGER NOT NULL, 
	confidence FLOAT NOT NULL, 
	status VARCHAR(2) NOT NULL, 
	semantic_similarity FLOAT NOT NULL, 
	keyword_overlap FLOAT NOT NULL, 
	unknown_term_ratio FLOAT NOT NULL, 
	matched_terms JSON NOT NULL, 
	evidence_text TEXT NOT NULL, 
	evidence JSON NOT NULL, 
	method VARCHAR(10) NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(question_id) REFERENCES exam_question (id) ON DELETE CASCADE, 
	FOREIGN KEY(topic_id) REFERENCES course_topic (id) ON DELETE CASCADE
);
CREATE TABLE source_file (
	id INTEGER NOT NULL, 
	course_id INTEGER NOT NULL, 
	kind VARCHAR(20) NOT NULL, 
	filename VARCHAR(400) NOT NULL, 
	sha256 VARCHAR(64) NOT NULL, 
	mime VARCHAR(100) NOT NULL, 
	size_bytes INTEGER NOT NULL, 
	stored_path VARCHAR(800) NOT NULL, 
	status VARCHAR(30) NOT NULL, 
	error TEXT NOT NULL, 
	page_count INTEGER NOT NULL, 
	extraction_summary JSON NOT NULL, 
	syllabus_version_id INTEGER, 
	uploaded_at DATETIME NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_course_file_hash UNIQUE (course_id, sha256), 
	FOREIGN KEY(course_id) REFERENCES course (id) ON DELETE CASCADE, 
	FOREIGN KEY(syllabus_version_id) REFERENCES syllabus_version (id) ON DELETE SET NULL
);
CREATE TABLE syllabus_version (
	id INTEGER NOT NULL, 
	course_id INTEGER NOT NULL, 
	label VARCHAR(120) NOT NULL, 
	is_current BOOLEAN NOT NULL, 
	effective_from_order FLOAT, 
	created_at DATETIME NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(course_id) REFERENCES course (id) ON DELETE CASCADE
);
CREATE INDEX ix_analysis_artifact_run_id ON analysis_artifact (run_id);
CREATE INDEX ix_analysis_run_course_id ON analysis_run (course_id);
CREATE INDEX ix_backtest_fold_run_id ON backtest_fold (run_id);
CREATE INDEX ix_course_topic_course_id ON course_topic (course_id);
CREATE INDEX ix_course_topic_parent_id ON course_topic (parent_id);
CREATE INDEX ix_course_topic_syllabus_version_id ON course_topic (syllabus_version_id);
CREATE INDEX ix_document_page_file_id ON document_page (file_id);
CREATE INDEX ix_exam_course_id ON exam (course_id);
CREATE INDEX ix_exam_question_exam_id ON exam_question (exam_id);
CREATE INDEX ix_exam_question_parent_id ON exam_question (parent_id);
CREATE INDEX ix_exam_section_exam_id ON exam_section (exam_id);
CREATE INDEX ix_mapping_topic ON question_topic_mapping (topic_id);
CREATE INDEX ix_model_result_run_id ON model_result (run_id);
CREATE INDEX ix_predicted_question_run_id ON predicted_question (run_id);
CREATE INDEX ix_prediction_run_id ON prediction (run_id);
CREATE INDEX ix_question_topic_mapping_question_id ON question_topic_mapping (question_id);
CREATE INDEX ix_source_file_course_id ON source_file (course_id);
CREATE INDEX ix_syllabus_version_course_id ON syllabus_version (course_id);
