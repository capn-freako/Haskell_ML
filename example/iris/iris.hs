-- Example use of `Haskell_ML.FCN` to categorize the Iris dataset.
--
-- Original author: David Banas <capn.freako@gmail.com>
-- Original date:   January 22, 2018
--
-- Copyright (c) 2018 David Banas; all rights reserved World wide.

{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import           Control.Arrow
import           Data.List
import           System.Random.Shuffle

import Haskell_ML.FCN
import Haskell_ML.Util

dataFileName :: String
dataFileName = "data/iris.csv"

main :: IO ()
main = do
  -- Read in the Iris data set. It contains an equal number of samples
  -- for all 3 classes of iris.
  putStrLn "Reading in data..."
  samps    <- readIrisData dataFileName

  -- Make field values uniform over [0,1].
  let samps' = mkSmplsUniform samps

  -- Split according to class, so we can keep equal representation
  -- throughout.
  let samps1 = filter ((== Setosa)     . snd) samps'
      samps2 = filter ((== Versicolor) . snd) samps'
      samps3 = filter ((== Virginica)  . snd) samps'

      -- Replace attributes record w/ feature vector.
      samps1' = map (first attributeToVector) samps1
      samps2' = map (first attributeToVector) samps2
      samps3' = map (first attributeToVector) samps3

      -- Replace iris type w/ one-hot vector.
      samps1'' = map (second irisTypeToVector) samps1'
      samps2'' = map (second irisTypeToVector) samps2'
      samps3'' = map (second irisTypeToVector) samps3'

  -- Shuffle samples.
  shuffled1 <- shuffleM samps1''
  shuffled2 <- shuffleM samps2''
  shuffled3 <- shuffleM samps3''

  -- Split into training/testing groups.
  -- We calculate separate lengths, even though we expect all 3 to be
  -- the same, just for safety's sake.
  let len1 = length shuffled1
      len2 = length shuffled2
      len3 = length shuffled3

      numTrn1 = len1 * 80 `div` 100
      numTrn2 = len2 * 80 `div` 100
      numTrn3 = len3 * 80 `div` 100

      -- training data
      trn1 = take numTrn1 shuffled1
      trn2 = take numTrn2 shuffled2
      trn3 = take numTrn3 shuffled3

      -- test data
      tst1 = drop numTrn1 shuffled1
      tst2 = drop numTrn2 shuffled2
      tst3 = drop numTrn3 shuffled3

      -- Reassemble into single training/testing sets.
      trn = trn1 ++ trn2 ++ trn3
      tst = tst1 ++ tst2 ++ tst3

  -- Reshuffle.
  trnShuffled <- shuffleM trn
  tstShuffled <- shuffleM tst

  putStrLn "Done."

  -- Ask user for internal network structure.
  putStrLn "Please, enter a list of integers specifying the width"
  putStrLn "of each hidden layer you want in your network."
  putStrLn "For instance, entering '[2, 4]' will give you a network"
  putStrLn "with 2 hidden layers:"
  putStrLn " - one (closest to the input layer) with 2 output nodes, and"
  putStrLn " - one with 4 output nodes."
  hs <- readLn
  n  <- randNet hs
  putStrLn "Great! Now, enter your desired learning rate."
  putStrLn "(Should be a decimal floating point value in (0,1)."
  rate <- readLn
  let (n', (accs, diffs))  = trainNTimes 60 rate n trnShuffled
      res = runNet n' $ map fst tstShuffled
      ref = map snd tstShuffled
  putStrLn $ "Test accuracy: " ++ (show $ classificationAccuracy res ref)

  putStrLn "Training accuracy:"
  putStrLn $ asciiPlot accs

  putStrLn "Average variance in first layer weights:"
  putStrLn $ asciiPlot $ map (calcMeanList . map (\x -> x*x) . head . fst) diffs
  putStrLn "Average variance in second layer weights:"
  putStrLn $ asciiPlot $ map (calcMeanList . map (\x -> x*x) . head . tail . fst) diffs
  putStrLn "Average variance in first layer biases:"
  putStrLn $ asciiPlot $ map (calcMeanList . map (\x -> x*x) . head . snd) diffs
  putStrLn "Average variance in second layer biases:"
  putStrLn $ asciiPlot $ map (calcMeanList . map (\x -> x*x) . head . tail . snd) diffs

