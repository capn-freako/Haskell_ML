-- Example use of `ConCat` to categorize the Iris dataset.
--
-- Original author: David Banas <capn.freako@gmail.com>
-- Original date:   February 21, 2018
--
-- Copyright (c) 2018 David Banas; all rights reserved World wide.
--
-- Note: this code started out as a direct copy of iris.hs.

{-# OPTIONS_GHC -Wall #-}
{-# OPTIONS_GHC -Wno-unused-imports #-}
{-# OPTIONS_GHC -Wno-missing-signatures #-}

{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators #-}

module Main where

import Prelude hiding (zipWith, zip)

import qualified GHC.Generics  -- To avoid coercion hole errors from ConCat.Plugin.

import           GHC.Generics ((:*:)(..))
import           Control.Arrow
import           Control.Monad
import           Data.Foldable (fold)
import           Data.Key     (Zip(..))
import           Data.List    hiding (zipWith, zip)
import           System.Random.Shuffle

import ConCat.Deep
import ConCat.Misc     (R)
import ConCat.Rebox    ()  -- Necessary for reboxing rules to fire
import ConCat.AltCat   ()  -- Necessary, but I've forgotten why.

import Haskell_ML.Util
import Haskell_ML.Classify.Classifiable
import Haskell_ML.Classify.Iris

-- Define the network and its parameter type.
-- type PType = ((V 10 --+ V 3) :*: (V 4 --+ V 10)) R
-- pfunc        = lr2  -- Defined in ConCat.Deep.
type PType = ((V 5 --+ V 3) :*: (V 10 --+ V 5) :*: (V 4 --+ V 10)) R
pfunc      = lr3'  -- Defined in ConCat.Deep.

dataFileName :: String
dataFileName = "data/iris.csv"

main :: IO ()
main = do
  rate   <- prompt "Learning rate?"
  epochs <- prompt "Number of epochs?"

  -- Read in the Iris data set. It contains an equal number of samples
  -- for all 3 classes of iris.
  putStrLn "Reading in data..."
  (samps :: [Sample Iris]) <- readClassifiableData dataFileName

  -- Shuffle samples.
  shuffled <- shuffleM samps

  -- Perform the following operations, in order:
  -- - Make attribute values uniform over [0,1].
  -- - Split samples according to class.
  -- - Split each class into training/testing sets.
  let splitV = fmap (splitTrnTst 80)
               . splitClassifiableData
               . uncurry zip
               . first mkAttrsUniform
               . unzip
               . map (first attrToVec)

  -- Gather up the training/testing sets into two lists and reshuffle.
  let (trn, tst) = fold (splitV shuffled)
  trnShuffled <- shuffleM trn
  tstShuffled <- shuffleM tst

  putStrLn "Done."

  -- Create random parameters, train them, and test the resultant accuracy.
  let ps         = (\x -> 2 * x - 1) <$> randF 1
      ps'        = trainNTimes epochs rate pfunc ps trnShuffled
      (res, ref) = unzip $ map (first (pfunc (last ps'))) tstShuffled
      accs       = map (\p -> uncurry classificationAccuracy $ unzip $ map (first (pfunc p)) trnShuffled) ps'
      diffs      = (zipWith.zipWith) (-) (tail ps') ps' :: [PType]

  putStrLn $ "Test accuracy: " ++ show (classificationAccuracy res ref)

  -- Plot the evolution of the training accuracy.
  putStrLn "Training accuracy:"
  putStrLn $ asciiPlot accs

  -- Plot the evolution of the weights and biases.
  putStrLn $ unlines $ showPart getWeights "weights" diffs
  putStrLn $ unlines $ showPart getBiases  "biases"  diffs

