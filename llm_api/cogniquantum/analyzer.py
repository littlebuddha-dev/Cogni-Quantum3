# /llm_api/cogniquantum/analyzer.py
# タイトル: Multi-Language and Edge-Aware Complexity Analyzer (Corrected and Enhanced)
# 役割: 複雑性分析ロジックを修正し、日本語のような非スペース区切り言語でもNLP分析が正しくトリガーされるようにする。

import logging
import spacy
from typing import Tuple, Optional, Dict, Any

try:
    from langdetect import detect, LangDetectException
except ImportError:
    logging.warning("langdetect ライブラリがインストールされていません。言語検出機能が無効化されます。")
    def detect(text):
        return 'en'
    class LangDetectException(Exception):
        pass

from .enums import ComplexityRegime
from .learner import ComplexityLearner

logger = logging.getLogger(__name__)

class AdaptiveComplexityAnalyzer:
    """
    プロンプトの言語を自動検出し、その言語に最適化された複雑性分析を行う。
    Edgeモードに対応し、リソース消費を抑制する。
    """
    def __init__(self, learner: Optional[ComplexityLearner] = None):
        self.learner = learner
        self.nlp_models: Dict[str, Any] = {}
        self.keyword_sets = {
            'en': {
                'conditional': ['if', 'when', 'unless', 'provided', 'given'],
                'hierarchy': ['first', 'second', 'then', 'next', 'finally', 'step'],
                'constraint': ['must', 'cannot', 'should not', 'requires', 'constraint'],
                'math': ['calculate', 'solve', 'equation', 'algorithm', 'optimization'],
                'planning': ['plan', 'strategy', 'design', 'organize', 'coordinate'],
                'analysis': ['analyze', 'compare', 'evaluate', 'assess', 'consider'],
            },
            'ja': {
                'conditional': ['場合', 'とき', 'たら', 'れば', 'なら', 'もし'],
                'hierarchy': ['まず', '次に', 'そして', '最後に', '第一に', '第二に', 'ステップ'],
                'constraint': ['必要', '必須', 'ならない', 'べき', '制約', '条件'],
                'math': ['計算', '解く', '方程式', 'アルゴリズム', '最適化'],
                'planning': ['計画', '戦略', '設計', '整理', '調整'],
                'analysis': ['分析', '比較', '評価', '検討', '考察'],
            }
        }
        self.spacy_model_map = {
            'en': 'en_core_web_sm',
            'ja': 'ja_core_news_sm',
            'de': 'de_core_news_sm',
            'es': 'es_core_news_sm',
            'fr': 'fr_core_news_sm',
        }
    
    def analyze_complexity(self, prompt: str, mode: str = 'adaptive') -> Tuple[float, ComplexityRegime]:
        """
        多言語とEdgeモードに対応した複雑性分析。
        """
        if mode == 'edge':
            logger.info("エッジモードのため、軽量なキーワード分析を実行し、低複雑性レジームに固定します。")
            return 10.0, ComplexityRegime.LOW

        if self.learner:
            suggestion = self.learner.get_suggestion(prompt)
            if suggestion:
                logger.info(f"学習済みの提案が見つかりました: 複雑性レジームを '{suggestion.value}' に設定します。")
                if suggestion == ComplexityRegime.LOW: return 15.0, suggestion
                if suggestion == ComplexityRegime.MEDIUM: return 50.0, suggestion
                if suggestion == ComplexityRegime.HIGH: return 85.0, suggestion

        lang = self._detect_language(prompt)
        
        nlp = self._get_spacy_model(lang)
        
        # 文字数でも判定するように改善（トークン化以前の簡易チェック）
        should_use_nlp = (len(prompt) > 30) if nlp else False
        
        if should_use_nlp:
            try:
                doc = nlp(prompt)
                if len(doc) > 5: # トークン数が5より大きい場合にNLP分析を実行
                    logger.info(f"'{lang}'言語のNLPベース高度分析を実行します。")
                    complexity_score = self._nlp_enhanced_analysis(doc)
                else:
                    logger.info(f"プロンプトのトークン数が少ないため、'{lang}'言語のキーワードベース分析を実行します。")
                    complexity_score = self._keyword_based_analysis(prompt, lang)
            except Exception as e:
                logger.warning(f"NLP分析でエラーが発生しました: {e}。キーワードベース分析にフォールバックします。")
                complexity_score = self._keyword_based_analysis(prompt, lang)
        else:
            logger.info(f"NLPモデルが利用できないか、プロンプトが短いため、'{lang}'言語のキーワードベース分析を実行します。")
            complexity_score = self._keyword_based_analysis(prompt, lang)
        
        logger.info(f"算出された複雑性スコア: {complexity_score:.2f}")

        if complexity_score < 30:
            regime = ComplexityRegime.LOW
        elif complexity_score < 70:
            regime = ComplexityRegime.MEDIUM
        else:
            regime = ComplexityRegime.HIGH
        
        logger.info(f"決定された複雑性レジーム: {regime.value}")
        return complexity_score, regime

    def _detect_language(self, text: str) -> str:
        """プロンプトの言語を検出する。"""
        try:
            if len(text) > 20:
                lang = detect(text)
                logger.info(f"検出された言語: {lang}")
                return lang
            else:
                logger.info("プロンプトが短すぎるため、デフォルト言語（英語）を使用します。")
                return 'en'
        except (LangDetectException, Exception) as e:
            logger.warning(f"言語の検出に失敗しました: {e}。デフォルト言語（英語）を使用します。")
            return 'en'

    def _get_spacy_model(self, lang: str):
        """言語に応じたspaCyモデルをロードする。"""
        if lang in self.nlp_models:
            return self.nlp_models[lang]

        model_name = self.spacy_model_map.get(lang)
        if not model_name:
            logger.warning(f"言語 '{lang}' に対応するspaCyモデルが定義されていません。")
            return None
            
        try:
            if not spacy.util.is_package(model_name):
                logger.info(f"spaCyモデル '{model_name}' が見つかりません。利用可能なモデルがないためスキップします。")
                self.nlp_models[lang] = None
                return None
            
            nlp = spacy.load(model_name)
            logger.info(f"spaCyモデル '{model_name}' のロードに成功しました。")
            self.nlp_models[lang] = nlp
            return nlp
        except (ImportError, OSError, SystemExit) as e:
            logger.warning(
                f"spaCyモデル '{model_name}' のロードに失敗しました: {e}\n"
                f"高度な分析を有効にするには、手動でインストールしてください: python -m spacy download {model_name}"
            )
            self.nlp_models[lang] = None
            return None

    def _keyword_based_analysis(self, prompt: str, lang: str) -> float:
        """言語に応じたキーワードセットを使用して複雑性を分析する。"""
        keywords = self.keyword_sets.get(lang, self.keyword_sets['en'])
        prompt_lower = prompt.lower()
        
        # 日本語の場合は単語分割を調整
        if lang == 'ja':
            # 日本語の場合、スペース区切りでの単語数計算は不正確なので文字数ベースで調整
            length_score = min(len(prompt) / 50.0, 40)  # 50文字で1スコア
        else:
            length_score = min(len(prompt.split()) / 5.0, 40)

        structural_complexity = 0
        structural_complexity += sum(prompt_lower.count(p) for p in keywords['conditional']) * 3
        if lang == 'ja':
            # 日本語の場合は文字列検索ベース
            structural_complexity += sum(1 for word in keywords['hierarchy'] if word in prompt_lower) * 2
        else:
            structural_complexity += sum(1 for word in prompt_lower.split() if word in keywords['hierarchy']) * 2
        structural_complexity += sum(prompt_lower.count(p) for p in keywords['constraint']) * 4
        structure_score = min(structural_complexity, 30)

        domain_complexity = 0
        if any(kw in prompt_lower for kw in keywords['math']): domain_complexity += 15
        if any(kw in prompt_lower for kw in keywords['planning']): domain_complexity += 20
        if any(kw in prompt_lower for kw in keywords['analysis']): domain_complexity += 15
        domain_score = min(domain_complexity, 30)

        weights = {'length': 0.2, 'structure': 0.4, 'domain': 0.4}
        total_score = (length_score * weights['length'] +
                       structure_score * weights['structure'] +
                       domain_score * weights['domain'])
        
        return min(max(total_score, 0), 100.0)

    def _nlp_enhanced_analysis(self, doc) -> float:
        """
        言語に依存しない、spaCyのDocオブジェクトを使用した高度な複雑性分析。
        """
        sentences = list(doc.sents)
        num_sentences = len(sentences)
        if num_sentences == 0: return 5.0
        avg_sent_length = len(doc) / num_sentences
        
        try:
            num_noun_chunks = len(list(doc.noun_chunks))
        except Exception:
            # noun_chunksが利用できない場合の代替処理
            num_noun_chunks = len([token for token in doc if token.pos_ == 'NOUN'])
            
        syntactic_score = (num_sentences * 1.5) + (avg_sent_length * 0.5) + (num_noun_chunks * 1.0)
        normalized_syntactic = min(syntactic_score / 40.0, 1.0) * 100

        num_entities = len(doc.ents)
        unique_entity_labels = len(set(ent.label_ for ent in doc.ents)) if doc.ents else 0
        entity_score = (num_entities * 2.0) + (unique_entity_labels * 3.0)
        content_words = {token.lemma_.lower() for token in doc if token.pos_ not in ['PUNCT', 'SPACE', 'SYM', 'NUM'] and not token.is_stop}
        lexical_diversity_score = len(content_words) * 0.2
        lexical_score = entity_score + lexical_diversity_score
        normalized_lexical = min(lexical_score / 50.0, 1.0) * 100

        cognitive_keywords = {'compare', 'contrast', 'analyze', 'evaluate', 'synthesize', 'create', 'argue', 'derive', 'prove', '比較', '対比', '分析', '評価', '統合', '創造', '議論', '導出', '証明'}
        cognitive_lemmas = {token.lemma_.lower() for token in doc if token.pos_ == 'VERB'}
        cognitive_demand_score = len(cognitive_keywords.intersection(cognitive_lemmas)) * 10
        
        # 疑問詞の検出を改善
        question_patterns = {'why', 'how', 'what', 'when', 'where', 'who', 'なぜ', 'どのように', 'なに', 'いつ', 'どこ', 'だれ'}
        question_words = {token.text.lower() for token in doc if token.text.lower() in question_patterns}
        if question_words:
            cognitive_demand_score += 15 if any(w in question_words for w in ['why', 'how', 'なぜ', 'どのように']) else 5
            
        normalized_cognitive = min(cognitive_demand_score / 30.0, 1.0) * 100

        weights = {'syntactic': 0.40, 'lexical': 0.35, 'cognitive': 0.25}
        total_score = (normalized_syntactic * weights['syntactic'] +
                       normalized_lexical * weights['lexical'] +
                       normalized_cognitive * weights['cognitive'])

        logger.debug(
            f"NLP Analysis Scores (Normalized): Syntactic={normalized_syntactic:.2f}, "
            f"Lexical={normalized_lexical:.2f}, Cognitive={normalized_cognitive:.2f}"
        )
        return min(max(total_score, 0), 100.0)