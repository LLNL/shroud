/*
 * Copyright Shroud Project Developers. See LICENSE file for details.
 *
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

#ifndef ENUM_H
#define ENUM_H

enum Color {
  RED = 10,
  BLUE,
  WHITE
};

enum val {
  a1,
  b1 = 3,
  c1,
  d1 = b1 - a1,
  e1 = d1,
  f1,
  g1,
  h1 = 100,
};

int convert_to_int(enum Color in);
enum Color returnEnum(enum Color in);
void returnEnumOutArg(enum Color *out);
enum Color returnEnumInOutArg(enum Color *inout);

#endif // ENUM_H

