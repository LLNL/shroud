//
//
#include "scope.hpp"
#include <iostream>

int main(int argc, char *argv[])
{

  Color c1 = RED;
  auto c2 = ns1Enum::Color::RED;
  auto c3 = ns2Enum::Color::RED;
  auto c4 = cls1Enum::Color::RED;
  auto c5 = cls2Enum::Color::RED;
  auto c6 = ColorEnum::RED;

  std::cout << static_cast<int>(c1) << std::endl;
  std::cout << static_cast<int>(c2) << std::endl;
  std::cout << static_cast<int>(c3) << std::endl;
  std::cout << static_cast<int>(c4) << std::endl;
  std::cout << static_cast<int>(c5) << std::endl;
  std::cout << static_cast<int>(c6) << std::endl;

  return 0;
}
