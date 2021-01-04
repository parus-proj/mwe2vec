#ifndef TRACER_H_
#define TRACER_H_

#include "vocabulary.h"

#include <string>
#include <vector>
#include <map>
#include <numeric>
#include <cmath>
#include <fstream>



// запоминает, как изменялось расстояние между выбранными лекс.единицами в ходе обучения дистриб.модели
class Tracer
{
private:
  const size_t INVALID_IDX = std::numeric_limits<size_t>::max();
  const std::vector< std::pair<std::string, std::string> > TEST_DATA = {
      {"точка_зрения", "точка"},
      {"транспортное_средство", "средство"},
      {"средства_массовой_информации", "средство"},
      {"торговый_центр", "центр"},
      {"программное_обеспечение", "обеспечение"},
      {"принять_душ", "принять"},
      {"подзорная_труба", "труба"},
      {"подводная_лодка", "лодка"},
      {"населенный_пункт", "пункт"},
      {"крыша_поехала", "поехать"},
      {"земной_шар", "шар"},
      {"денежные_средства", "средство"},
      {"железная_дорога", "дорога"}
  };
public:
  void init(std::shared_ptr< CustomVocabulary > vocab)
  {
    for (auto& d : TEST_DATA)
    {
      size_t w1idx = vocab->word_to_idx(d.first);
      size_t w2idx = vocab->word_to_idx(d.second);
      if (w1idx == INVALID_IDX || w2idx == INVALID_IDX) continue;
      to_trace[w1idx] = w2idx;
    }
  }
  void run(/*size_t word1,*/ std::shared_ptr< CustomVocabulary > vocab, float* embeddings, size_t emb_size)
  {
//    auto it = to_trace.find(word1);
//    if (it == to_trace.end())
//      return;
//    size_t word2 = it->second;
//    float sim = cosm(word1, word2, embeddings, emb_size);
//    traces[ std::make_pair(word1, word2) ].push_back(sim);
    static size_t cnt = 0;
    if (cnt++ % 25000 != 0) return;
    for (auto& d : TEST_DATA)
    {
      size_t w1idx = vocab->word_to_idx(d.first);
      size_t w2idx = vocab->word_to_idx(d.second);
      if (w1idx == INVALID_IDX || w2idx == INVALID_IDX) continue;
      float sim = cosm(w1idx, w2idx, embeddings, emb_size);
      traces[ std::make_pair(w1idx, w2idx) ].push_back(sim);
    }
  }
  float cosm(size_t word1, size_t word2, float* embeddings, size_t emb_size)
  {
    // вычисляем косинусную меру сходства
    float* w1_Offset = embeddings + word1*emb_size;
    float* w2_Offset = embeddings + word2*emb_size;
    float sim = std::inner_product(w1_Offset, w1_Offset+emb_size, w2_Offset, 0.0);
    sim /= std::sqrt( std::inner_product(w1_Offset, w1_Offset+emb_size, w1_Offset, 0.0) );
    sim /= std::sqrt( std::inner_product(w2_Offset, w2_Offset+emb_size, w2_Offset, 0.0) );
    return sim;
  }
  void save(std::shared_ptr< CustomVocabulary > vocab)
  {
    for (auto& t : traces)
    {
      const std::string& w1 = vocab->idx_to_data(t.first.first).word;
      const std::string& w2 = vocab->idx_to_data(t.first.second).word;
      std::string trace_name = "trace-"+w1 + "-vs-" + w2 + ".csv";
      std::ofstream ofs(trace_name.c_str());
      for (auto v : t.second)
        ofs << v << std::endl;
    }
  }
private:
  std::map<std::pair<size_t,size_t>, std::vector<float>> traces;
  std::map<size_t, size_t> to_trace;
};


#endif /* TRACER_H_ */
