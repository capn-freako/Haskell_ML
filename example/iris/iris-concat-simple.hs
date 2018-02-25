-- Simplified test case for debugging plug-in trouble.
--
-- Original author: David Banas <capn.freako@gmail.com>
-- Original date:   February 25, 2018
--
-- Copyright (c) 2018 David Banas; all rights reserved World wide.
--
-- Note: this code started out as a direct copy of iris-concat.hs.

{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}

module Main where

import           GHC.Generics ((:*:)(..))
import qualified Data.Vector.Sized as VS

import ConCat.Deep
import ConCat.Misc     (R)
import ConCat.Rebox    () -- necessary for reboxing rules to fire
import ConCat.AltCat   ()

import Haskell_ML.Util ( randF )

type V = VS.Vector

main :: IO ()
main = do
  -- Use 2-layer network, from `ConCat.Deep`.
  let net  = randF 1 :: ((V 10 --+ V 3) :*: (V 4 --+ V 10)) R
      net' = steps 0.1 lr2 [(VS.replicate 0.0 :: V 4 R, VS.replicate 0.0 :: V 3 R)] net
  putStrLn $ show net'

