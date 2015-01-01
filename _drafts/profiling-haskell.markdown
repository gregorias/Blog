---
layout: post
title:  "Profiling Haskell - fixing time and space leaks"
date:   2014-12-29 22:00:00
tags: blog programming haskell
---
In an [earlier post] I have explained the rationale behind some data types in
Pentago game implementation. Here I will show how I have used Haskell's
profiling tools to uncover, understand and fix serious time and space leaks
inside this application. 

Time leak
---------

Space leak
----------
