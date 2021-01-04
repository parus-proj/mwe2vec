#ifndef VOCABS_BUILDER_H_
#define VOCABS_BUILDER_H_

#include "conll_reader.h"
#include "mwe_vocabulary.h"
#include "original_word2vec_vocabulary.h"

#include <memory>
#include <string>
#include <cstring>       // for std::strerror
#include <vector>
#include <unordered_map>
#include <set>
#include <fstream>

// Класс, хранящий данные по чтению обучающих данных и выводящий прогресс-сообщения
class StatHelper
{
public:
  void calc_sentence(size_t cnt)
  {
    tokens_processed += cnt; // статистику ведём и по некорректным предложениям
    tokens_dbg_counter += cnt;
    if (tokens_dbg_counter >= 100000)
    {
      tokens_dbg_counter %= 100000;
      if (tokens_processed >= 1000000)
        std::cout << '\r' << (tokens_processed / 1000000) << " M        ";
      else
        std::cout << '\r' << (tokens_processed / 1000) << " K        ";
      std::cout.flush();
    }
    if (cnt > 0)
      sentence_processed++;
  }
  void inc_sr_fils()
  {
    ++sr_fails_cnt;
  }
  void output_stat()
  {
    if ( sr_fails_cnt > 0)
      std::cerr << "Sentence reading fails count: " << sr_fails_cnt << std::endl;
    std::cout << "Sentences count: " << sentence_processed << std::endl;
    std::cout << "Tokens count: " << tokens_processed << std::endl;
    std::cout << std::endl;
  }
private:
  uint64_t sr_fails_cnt = 0;    // количество ошибок чтения предложений (предложений, содержащих хотя бы одну некорректную запись)
  uint64_t sentence_processed = 0;
  uint64_t tokens_processed = 0;
  size_t   tokens_dbg_counter = 0;
};


// Класс, обеспечивающие создание словарей (-task vocab)
class VocabsBuilder
{
private:
  typedef std::vector< std::vector<std::string> > SentenceMatrix;
  typedef std::unordered_map<std::string, uint64_t> VocabMapping;
  typedef std::shared_ptr<VocabMapping> VocabMappingPtr;
  typedef std::unordered_map<std::string, std::map<std::string, size_t>> Token2LemmasMap;
  typedef std::shared_ptr<Token2LemmasMap> Token2LemmasMapPtr;
public:
  // построение всех словарей
  bool build_vocabs(const std::string& conll_fn, const std::string& mwe_fn,
                    const std::string& voc_m_fn, const std::string& voc_p_fn, const std::string& voc_t_fn,
                    const std::string& voc_tm_fn, const std::string& voc_d_fn,
                    size_t limit_m, size_t limit_p, size_t limit_t, size_t limit_d,
                    size_t ctx_vocabulary_column_d, bool use_deprel)
  {
    // Проход 1: строим главный словарь (включая словосочетания)

    bool succ = build_main_vocab_only(conll_fn, mwe_fn, voc_m_fn, limit_m);
    if ( !succ ) return false;

    // Проход2: строим остальные словари уже с учётом того, какие именно словосочетания преодолели частотный порог основного словаря

    // создаём справочник словосочетаний
    std::shared_ptr< OriginalWord2VecVocabulary > v_main = std::make_shared<OriginalWord2VecVocabulary>();
    if ( !v_main->load(voc_m_fn) )
      return false;
    std::shared_ptr< MweVocabulary > v_mwe = std::make_shared<MweVocabulary>();
    if ( !v_mwe->load(mwe_fn, v_main) )
      return false;

    // открываем файл с тренировочными данными
    FILE *conll_file = fopen(conll_fn.c_str(), "rb");
    if ( conll_file == nullptr )
    {
      std::cerr << "Train-file open: error: " << std::strerror(errno) << std::endl;
      return false;
    }

    // создаем контейнеры для словарей
    VocabMappingPtr vocab_lemma_proper = std::make_shared<VocabMapping>();
    VocabMappingPtr vocab_token = std::make_shared<VocabMapping>();
    Token2LemmasMapPtr token2lemmas_map = std::make_shared<Token2LemmasMap>();
    VocabMappingPtr vocab_dep = std::make_shared<VocabMapping>();

    // в цикле читаем предложения из CoNLL-файла и извлекаем из них информацию для словарей
    SentenceMatrix sentence_matrix;
    sentence_matrix.reserve(5000);
    StatHelper stat;
    while ( !feof(conll_file) )
    {
      bool succ = ConllReader::read_sentence(conll_file, sentence_matrix);
      stat.calc_sentence(sentence_matrix.size());
      if (!succ)
      {
        stat.inc_sr_fils();
        continue;
      }
      if (sentence_matrix.size() == 0)
        continue;
      apply_patches(sentence_matrix); // todo: УБРАТЬ!  временный дополнительный корректор для борьбы с "грязными данными" в результатах лемматизации
      v_mwe->put_phrases_into_sentence(sentence_matrix);
      process_sentence_lemmas_proper(vocab_lemma_proper, sentence_matrix);
      process_sentence_tokens(vocab_token, token2lemmas_map, sentence_matrix);
      process_sentence_dep_ctx(vocab_dep, sentence_matrix, ctx_vocabulary_column_d, use_deprel);
    }
    fclose(conll_file);
    std::cout << std::endl;
    stat.output_stat();

    // сохраняем словари в файлах
    std::cout << "Save lemmas proper-names vocabulary..." << std::endl;
    save_vocab(vocab_lemma_proper, limit_p, voc_p_fn);
    std::cout << "Save tokens vocabulary..." << std::endl;
    save_vocab(vocab_token, limit_t, voc_t_fn, token2lemmas_map, voc_tm_fn);
    std::cout << "Save dependency contexts vocabulary..." << std::endl;
    save_vocab(vocab_dep, limit_d, voc_d_fn);
    return true;
  } // method-end
private:
  // функция построения и сохранения главного словаря
  // выполняется отдельно, т.к. необходимо выяснить частоты словосочетаний (какие из них преодолевают частотный порог главного словаря и будут преобразовываться)
  bool build_main_vocab_only(const std::string& conll_fn, const std::string& mwe_fn, const std::string& voc_m_fn, size_t limit_m)
  {
    // создаём справочник словосочетаний
    std::shared_ptr< MweVocabulary > v_mwe = std::make_shared<MweVocabulary>();
    if ( !v_mwe->load(mwe_fn) )
      return false;
    // открываем файл с тренировочными данными
    FILE *conll_file = fopen(conll_fn.c_str(), "rb");
    if ( conll_file == nullptr )
    {
      std::cerr << "Train-file open: error: " << std::strerror(errno) << std::endl;
      return false;
    }
    // создаем контейнер для словаря
    VocabMappingPtr vocab_lemma_main = std::make_shared<VocabMapping>();
    // в цикле читаем предложения из CoNLL-файла и извлекаем из них информацию для словаря
    SentenceMatrix sentence_matrix;
    sentence_matrix.reserve(5000);
    StatHelper stat;
    while ( !feof(conll_file) )
    {
      bool succ = ConllReader::read_sentence(conll_file, sentence_matrix);
      stat.calc_sentence(sentence_matrix.size());
      if (!succ)
      {
        stat.inc_sr_fils();
        continue;
      }
      if (sentence_matrix.size() == 0)
        continue;
      apply_patches(sentence_matrix); // todo: УБРАТЬ!  временный дополнительный корректор для борьбы с "грязными данными" в результатах лемматизации
      v_mwe->put_phrases_into_sentence(sentence_matrix);
      process_sentence_lemmas_main(vocab_lemma_main, sentence_matrix);
    }
    fclose(conll_file);
    std::cout << std::endl;
    stat.output_stat();
    // сохраняем словарь в файл
    std::cout << "Save lemmas main vocabulary..." << std::endl;
    erase_main_stopwords(vocab_lemma_main); // todo: УБРАТЬ!  временный дополнительный фильтр для борьбы с "грязными данными" в результатах морфологического анализа
    save_vocab(vocab_lemma_main, limit_m, voc_m_fn);
    return true;
  } // method-end
  // проверка, является ли токен собственным именем
  bool isProperName(const std::string& feats)
  {
    return feats.length() >=2 && feats[0] == 'N' && feats[1] == 'p';
  } // method-end
  // проверка, является ли токен стоп-словом для основного словаря (служит для исправления ошибок в разметке собственных имен)
  void erase_main_stopwords(VocabMappingPtr vocab)
  {
    static bool isListLoaded = false;
    std::set<std::string> stoplist;
    if (!isListLoaded)
    {
      isListLoaded = true;
      std::ifstream ifs("stopwords.common_nouns");
      std::string line;
      while ( std::getline(ifs, line).good() )
        stoplist.insert(line);
    }
    std::cout << "  stopwords reduce (main)" << std::endl;
    auto it = vocab->begin();
    while (it != vocab->end())    //TODO: в C++20 заменить на std::erase_if (https://en.cppreference.com/w/cpp/container/map/erase_if)
    {
      if (stoplist.find(it->first) == stoplist.end())
        ++it;
      else
        it = vocab->erase(it);
    }
  }
  void apply_patches(SentenceMatrix& sentence)
  {
    for ( auto& token : sentence )
      if ( is_punct__patch(token[2]) )
        token[7] = "PUNC";
  }
  void process_sentence_lemmas_main(VocabMappingPtr vocab, const SentenceMatrix& sentence)
  {
    for ( auto& token : sentence )
    {
      if (token[7] == "PUNC")  // знаки препинания в основной словарь не включаем (они обрабатываются особо)
        continue;
      if ( isProperName(token[5]) )
        continue;
      if ( token[2] == "_" ) // символ отсутствия значения в conll
        continue;
      auto& word = token[2];
      auto it = vocab->find( word );
      if (it == vocab->end())
        (*vocab)[word] = 1;
      else
        ++it->second;
    }
  } // method-end
  void process_sentence_lemmas_proper(VocabMappingPtr vocab, const SentenceMatrix& sentence)
  {
    for (auto& token : sentence)
    {
      if (token[7] == "PUNC")  // знаки препинания в словарь собственных имен не включаем
        continue;
      if ( !isProperName(token[5]) )
        continue;
      if ( token[2] == "_" ) // символ отсутствия значения в conll
        continue;
      auto& word = token[2];
      auto it = vocab->find( word );
      if (it == vocab->end())
        (*vocab)[word] = 1;
      else
        ++it->second;
    }
  } // method-end
  void process_sentence_tokens(VocabMappingPtr vocab, Token2LemmasMapPtr token2lemmas_map, const SentenceMatrix& sentence)
  {
    for ( auto& token : sentence )
    {
      if (token[7] == "PUNC")  // знаки препинания в словарь не включаем (они обрабатываются особо)
        continue;
      if ( token[1] == "_" || token[2] == "_" )   // символ отсутствия значения в conll
        continue;
      auto word = token[1];

      auto it = vocab->find( word );
      if (it == vocab->end())
        (*vocab)[word] = 1;
      else
        ++it->second;

      auto itt = (*token2lemmas_map)[word].find( token[2] );
      if ( itt == (*token2lemmas_map)[word].end() )
        (*token2lemmas_map)[word][token[2]] = 1;
      else
        ++itt->second;
    }
  } // method-end
  void process_sentence_dep_ctx(VocabMappingPtr vocab, const SentenceMatrix& sentence, size_t column, bool use_deprel)
  {
    for (auto& token : sentence)
    {
      if ( use_deprel )
      {
        if ( token[7] == "PUNC" )  // знаки препинания в словарь синтаксических контекстов не включаем
          continue;
        if ( token[column] == "_" || token[7] == "_" )  // символ отсутствия значения в conll
          continue;
        size_t parent_token_no = 0;
        try {
          parent_token_no = std::stoi(token[6]);
        } catch (...) {
          parent_token_no = 0; // если конвертирование неудачно, считаем, что нет родителя
        }
        if ( parent_token_no == 0 )
          continue;
        auto& parent = sentence[ parent_token_no - 1 ];
        if ( parent[7] == "PUNC" ) // "контексты -- знаки препинания" нам не интересны
          continue;                // note: не посчитаем контекст вниз, но его и не нужно, т.к. это контекст знака пунктуации
        if ( parent[column] == "_" ) // символ отсутствия значения в conll
          continue;

        // рассматриваем контекст с точки зрения родителя в синтаксической связи
        auto ctx__from_head_viewpoint = token[column] + "<" + token[7];
        auto it_h = vocab->find( ctx__from_head_viewpoint );
        if (it_h == vocab->end())
          (*vocab)[ctx__from_head_viewpoint] = 1;
        else
          ++it_h->second;
        // рассматриваем контекст с точки зрения потомка в синтаксической связи
        auto ctx__from_child_viewpoint = parent[column] + ">" + token[7];
        auto it_c = vocab->find( ctx__from_child_viewpoint );
        if (it_c == vocab->end())
          (*vocab)[ctx__from_child_viewpoint] = 1;
        else
          ++it_c->second;
      }
      else
      {
        if ( token[7] == "PUNC" )   // знаки препинания в словарь синтаксических контекстов не включаем
          continue;
        if ( token[column] == "_" ) // символ отсутствия значения в conll
          continue;
        auto& word = token[column];
        auto it = vocab->find( word );
        if (it == vocab->end())
          (*vocab)[word] = 1;
        else
          ++it->second;
      } // if ( use_depre ) then ... else ...
    }
  } // method-end
  bool is_punct__patch(const std::string& word)
  {
    const std::set<std::string> puncts = { ".", ",", "!", "?", ":", ";", "…", "...", "--", "—", "–", "‒",
                                           "'", "ʼ", "ˮ", "\"", "«", "»", "“", "”", "„", "‟", "‘", "’", "‚", "‛",
                                           "(", ")", "[", "]", "{", "}", "⟨", "⟩" };
    if ( puncts.find(word) != puncts.end() )
      return true;
    else
      return false;
  }
  // редукция и сохранение словаря в файл
  void save_vocab(VocabMappingPtr vocab, size_t min_count, const std::string& file_name, Token2LemmasMapPtr t2l = nullptr, const std::string& tlm_fn = std::string())
  {
    // удаляем редкие слова (ниже порога отсечения)
    std::cout << "  min-count reduce" << std::endl;
    auto it = vocab->begin();
    while (it != vocab->end())    //TODO: в C++20 заменить на std::erase_if (https://en.cppreference.com/w/cpp/container/map/erase_if)
    {
      if (it->second >= min_count)
        ++it;
      else
        it = vocab->erase(it);
    }
    std::cout << "  resulting vocabulary size: " << vocab->size() << std::endl;
    // пересортируем в порядке убывания частоты
    std::multimap<uint64_t, std::string, std::greater<uint64_t>> revVocab;
    for (auto& record : *vocab)
      revVocab.insert( std::make_pair(record.second, record.first) );
    // сохраняем словарь в файл
    FILE *fo = fopen(file_name.c_str(), "wb");
    for (auto& record : revVocab)
      fprintf(fo, "%s %lu\n", record.second.c_str(), record.first);
    fclose(fo);
    // сохраняем мэппинг из токенов в леммы (если в функцию передана соответствующая структура)
    // сохранение происходит с учётом отсечения по частотному порогу и переупорядочения (как в самом словаре токенов)
    if (t2l)
    {
      std::ofstream t2l_fs( tlm_fn.c_str() );
      for (auto& record : revVocab)
      {
        auto it = t2l->find(record.second);
        if (it == t2l->end()) continue;
        t2l_fs << record.second;
        for (auto& lemma : it->second)
          t2l_fs << " " << lemma.first << " " << lemma.second;
        t2l_fs << "\n";
      }
    }
  } // method-end
};


#endif /* VOCABS_BUILDER_H_ */
