// Mirrors the tribunal API JSON. Timestamps are ISO strings.

export interface User { id: string; email: string; is_admin: boolean }

export interface Subject { id: string; name: string; role: string }

export type DocumentStatus = 'queued' | 'processing' | 'ready' | 'failed';

export interface DocumentInfo {
  id: string;
  filename: string;
  status: DocumentStatus;
  error: string | null;
  page_count: number | null;
  created_at: string;
  ready_at: string | null;
  searchable_seconds: number | null;
  excluded_pages: number[];
}

export interface UploadResult { document: DocumentInfo; job_id: string }

export interface Topic {
  id: string;
  title: string;
  document_id: string | null;
  page_start: number | null;
  page_end: number | null;
  source: string;
}

export interface Passage {
  id: string;
  document_id: string;
  filename: string;
  page_number: number;
  text: string;
  url: string;
}

export interface Choice { key: string; label: string }

export interface Options {
  question_types: Choice[];
  bloom_levels: Choice[];
  difficulty_levels: Choice[];
  default_ocr_languages: string[];
}

export interface TemplateSection {
  title: string;
  question_type: string;
  count: number;
  marks_each: number;
  bloom: string;
  difficulty: string;
  answer_any?: number | null;
  parts?: number[] | null; // marks per part
}

export interface TemplateSpec {
  sections: TemplateSection[];
  total_marks: number;
  instructions?: string[];
}

export interface Template { id: string; name: string; version: number; spec: TemplateSpec }

export interface PaperSummary {
  id: string;
  title: string;
  status: string;
  created_at: string;
  template: { name: string; version: number };
}

export type SlotStatus = 'pending' | 'running' | 'drafted' | 'needs_review' | 'evidence_gap' | 'failed' | 'approved';

export interface RubricItem { criterion: string; marks: number }

export interface DraftPart {
  label: string;
  question: string;
  answer: string;
  rubric: RubricItem[];
  citations: string[];
  options: string[] | null;
  correct_option: number | null;
}

export type VerificationStatus = 'pass' | 'mismatch' | 'unsupported' | 'invalid' | 'unavailable';

// The solver's own fields (expression, option_index) are kept when present.
export interface Verification {
  status: VerificationStatus;
  detail: string;
  model?: string;
  prompt_version?: string;
  expression?: string;
  option_index?: number;
}

export interface RevisionContent {
  question: string;
  answer: string;
  parts?: DraftPart[] | null;
  rubric: RubricItem[];
  citations: string[];
  options: string[] | null;
  correct_option: number | null;
  verification?: Verification;
}

export interface Disputed { verdict: string; reasons: string[]; model: string }

export interface Review {
  verdict?: 'pass' | 'revise' | 'faculty_review';
  reasons?: string[];
  model?: string;
  disputed?: Disputed;
  verification?: Verification;
}

export interface Revision {
  id: string;
  number: number;
  author: string;
  content: RevisionContent;
  review: Review | null;
  model: string | null;
  prompt_version: string | null;
  approved_by: string | null;
  approved_at: string | null;
}

export interface Citation { passage_id: string; document_id: string; page_number: number; url: string }

export interface Slot {
  id: string;
  number: number;
  section: string;
  question_type: string;
  marks: number;
  topic_title: string;
  status: SlotStatus;
  status_detail: string | null;
  revision: Revision | null;
  citations: Citation[];
}

export interface Paper {
  id: string;
  title: string;
  status: string; // queued, running, drafted or failed
  error: string | null;
  subject_id: string;
  template: { id: string; name: string; version: number };
  config_snapshot_id: string;
  document_ids: string[];
  tokens_used: number;
  timings: { first_question_seconds: number | null; total_seconds: number | null };
  slots: Slot[];
}

export interface ApprovalResult { slot_id: string; number: number; status: SlotStatus; revision: Revision }

export interface PaperInput {
  subject_id: string;
  template_id: string;
  topic_ids: string[];
  document_ids?: string[] | null;
  title: string;
}

export interface ExportPart {
  label: string;
  question: string;
  options?: string[] | null;
  answer?: string;
  rubric?: RubricItem[];
  citations?: Citation[];
}

export interface ExportQuestion {
  number: number;
  marks: number;
  question: string;
  options: string[] | null;
  parts?: ExportPart[];
  answer?: string;
  rubric?: RubricItem[];
  citations?: Citation[];
}

export interface ExportSection { title: string; answer_any: number | null; questions: ExportQuestion[] }

export interface ExportDoc {
  paper_id: string;
  title: string;
  subject: string;
  template: { name: string; version: number };
  total_marks: number;
  instructions: string[];
  sections: ExportSection[];
  approvals?: { number: number; revision_id: string; approved_by: string; approved_at: string }[];
}
