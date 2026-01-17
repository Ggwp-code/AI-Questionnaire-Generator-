export interface QuestionData {
  question: string;
  answer: string;
  explanation: string;
  options?: string[];
  verification_code?: string;
  difficulty_rating: string;
  source?: string;
  source_urls?: string[];
  source_pages?: number[];  // Page numbers from PDF
  source_filename?: string;  // PDF filename
  computed_answer?: string;
  answer_warning?: string;
  quality_score?: number;  // Critic score 0-10
}

export interface UploadedDocument {
  filename: string;
  path: string;
  chunks: number;
  hash: string;
}

export interface GenerationResponse {
  status: string;
  data: QuestionData;
  meta: {
    duration_seconds: number;
    engine: string;
  };
}

export interface IngestionStats {
  chunk_count: number;
  total_pages: number;
  success: boolean;
  error_message?: string;
  file_hash: string;
  processing_time_ms: number;
}

export interface UploadResponse {
  filename: string;
  status: string;
  ingestion: IngestionStats;
}

export interface LogEntry {
  id: string;
  timestamp: string;
  level: 'INFO' | 'WARNING' | 'ERROR' | 'SUCCESS';
  message: string;
}

// Topic Suggestions
export interface TopicSuggestion {
  topic: string;
  examples: string[];
}

// Paper Generator Types
export interface QuestionSpec {
  topic: string;
  question_type: 'short' | 'long' | 'mcq' | 'numerical';
  difficulty: 'Easy' | 'Medium' | 'Hard';
  marks: number;
}

export interface PaperSection {
  name: string;
  instructions?: string;
  questions: QuestionSpec[];
}

export interface PaperTemplate {
  title: string;
  subject: string;
  duration_minutes: number;
  total_marks: number;
  instructions: string[];
  sections: PaperSection[];
}

// Backend returns question data directly (not nested in spec)
export interface GeneratedQuestion {
  question_number: number;
  part_of_section: string;
  topic: string;
  question_type: 'short' | 'long' | 'mcq' | 'numerical';
  difficulty: 'Easy' | 'Medium' | 'Hard';
  marks: number;
  parts_marks: Array<{ part: string; marks: number; description: string }>;
  question_text: string;  // Backend uses question_text
  answer: string;
  explanation: string;
  verification_code?: string;
  tags?: string[];
}

export interface GeneratedSection {
  name: string;
  instructions?: string;
  questions: GeneratedQuestion[];
}

export interface GeneratedPaper {
  paper_id: string;
  title: string;
  subject: string;
  duration_minutes: number;
  total_marks: number;
  instructions: string[];
  sections: GeneratedSection[];
  generated_at: string;
}

export interface PaperCreateResponse {
  paper_id: string;
  status: string;
  template: PaperTemplate;
}

export interface PaperGenerateResponse {
  status: string;
  paper: GeneratedPaper;
  generation_time_seconds: number;
}