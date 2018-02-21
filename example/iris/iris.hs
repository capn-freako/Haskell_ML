-- Example use of `Haskell_ML.FCN` to categorize the Iris dataset.
--
-- Original author: David Banas <capn.freako@gmail.com>
-- Original date:   January 22, 2018
--
-- Copyright (c) 2018 David Banas; all rights reserved World wide.

{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import           Control.Monad
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

  -- Make field values uniform over [0,1] and split according to class,
  -- so we can keep equal representation throughout.
  let (samps1, samps2, samps3) = splitIrisData $ mkSmplsUniform samps

  -- Shuffle samples and split into training/testing groups.
  shuffled1 <- shuffleM samps1
  shuffled2 <- shuffleM samps2
  shuffled3 <- shuffleM samps3
  let [(trn1, tst1), (trn2, tst2), (trn3, tst3)] = map splitTrnTst [shuffled1, shuffled2, shuffled3]
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
  let (n', TrainEvo{..}) = trainNTimes 60 rate n trnShuffled
      res = runNet n' $ map fst tstShuffled
      ref = map snd tstShuffled
  putStrLn $ "Test accuracy: " ++ show (classificationAccuracy res ref)

  -- Plot the evolution of the training accuracy.
  putStrLn "Training accuracy:"
  putStrLn $ asciiPlot accs

  -- Plot the evolution of the weights and biases.
  let weights = zip [1::Int,2..] $ (transpose . map fst) diffs
      biases  = zip [1::Int,2..] $ (transpose . map snd) diffs
  forM_ weights $ \ (i, ws) -> do
    putStrLn $ "Average variance in layer " ++ show i ++ " weights:"
    putStrLn $ asciiPlot $ map (calcMeanList . map (\x -> x*x)) ws
  forM_ biases $ \ (i, bs) -> do
    putStrLn $ "Average variance in layer " ++ show i ++ " biases:"
    putStrLn $ asciiPlot $ map (calcMeanList . map (\x -> x*x)) bs

