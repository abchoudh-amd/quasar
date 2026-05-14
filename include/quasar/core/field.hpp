#pragma once

#include <cstddef>
#include <memory>

namespace quasar {

template <class T>
class Field {
 public:
  Field() noexcept = default;

  explicit Field(std::size_t n)
    : size_{n},
      data_{n == 0 ? std::unique_ptr<T[]>{} : std::make_unique<T[]>(n)} {}

  Field(const Field&)            = delete;
  Field& operator=(const Field&) = delete;
  Field(Field&&) noexcept            = default;
  Field& operator=(Field&&) noexcept = default;
  ~Field()                       = default;

  std::size_t size()  const noexcept { return size_; }
  bool        empty() const noexcept { return size_ == 0; }

  T*       data()       noexcept { return data_.get(); }
  const T* data() const noexcept { return data_.get(); }

  T&       operator[](std::size_t i)       noexcept { return data_[i]; }
  const T& operator[](std::size_t i) const noexcept { return data_[i]; }

  T*       begin()       noexcept { return data_.get(); }
  const T* begin() const noexcept { return data_.get(); }
  T*       end()         noexcept { return data_.get() + size_; }
  const T* end()   const noexcept { return data_.get() + size_; }

 private:
  std::size_t          size_{0};
  std::unique_ptr<T[]> data_{};
};

}  // namespace quasar
