#ifndef BALANCE_H_
#define BALANCE_H_

#include "vectors_model.h"

#include <memory>
#include <string>
#include <fstream>
#include <iostream>

// Балансировщик dep/assoc-частей модели
class Balancer
{
public:
  static void run( const std::string& model_fn, bool useTxtFmt, size_t dep_size, float a_ratio )
  {
    // 1. Загружаем модель
    VectorsModel vm;
    if ( !vm.load(model_fn, useTxtFmt) )
      return;

    // 2. Корректируем веса ассоциативной части векторов
    for (size_t a = 0; a < vm.vocab.size(); ++a)
      for (size_t b = dep_size; b < vm.emb_size; ++b)
        vm.embeddings[a * vm.emb_size + b] *= a_ratio;

    // 3. Сохраняем модель
    vm.save(model_fn, useTxtFmt);
  } // method-end
}; // class-decl-end


#endif /* BALANCE_H_ */
