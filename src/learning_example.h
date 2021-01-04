#ifndef LEARNING_EXAMPLE_H_
#define LEARNING_EXAMPLE_H_

#include <vector>


// структура, представляющая обучающий пример
struct LearningExample
{
  size_t word;                         // индекс слова
  std::vector<size_t> dep_context;     // индексы синтаксических контекстов
  std::vector<size_t> assoc_context;   // индексы ассоциативных контекстов
};


#endif /* LEARNING_EXAMPLE_H_ */
