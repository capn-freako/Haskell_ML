-- Example use of `Haskell_ML.FCN` to categorize the Iris dataset.
--
-- This version of the Iris example enforces 4-bit precision on activation
-- outputs, when running inference. Training is performed, as usual.
--
-- Original author: David Banas <capn.freako@gmail.com>
-- Original date:   January 22, 2018
--
-- Copyright (c) 2018 David Banas; all rights reserved World wide.

{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import           Control.Arrow (first, (***))
import           Control.Monad
import           Data.Foldable (fold)
import           Data.List
import qualified Data.Vector.Sized as VS
import           System.Random.Shuffle

import Haskell_ML.FCN
import Haskell_ML.Util                   hiding (classificationAccuracy)
import Haskell_ML.Classify.Classifiable
import Haskell_ML.Classify.Iris

dataFileName :: String
dataFileName = "data/iris.csv"

main :: IO ()
main = do
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
  let splitV = VS.map ( splitTrnTst 80
                      . map (toR *** toR)
                      ) $ (splitClassifiableData . uncurry zip . first mkAttrsUniform . unzip)
                        $ map (first attrToVec) shuffled

  -- Gather up the training/testing sets into two lists and reshuffle.
  let (trn, tst) = fold splitV
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
  net  <- randNet hs
  putStrLn "Great! Now, enter your desired learning rate."
  putStrLn "(Should be a decimal floating point value in (0,1)."
  rate <- readLn
  putStrLn "And, finally, the number of epochs you'd like to run."
  putStrLn "(Should be an integer between 120 and 10,000.)"
  epochs <- readLn
  let (n', TrainEvo{..}) = trainNTimes epochs rate net trnShuffled
      res = runNet' n' $ map fst tstShuffled
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
    putStrLn $ asciiPlot $ map (mean . map (\x -> x*x)) ws
  forM_ biases $ \ (i, bs) -> do
    putStrLn $ "Average variance in layer " ++ show i ++ " biases:"
    putStrLn $ asciiPlot $ map (mean . map (\x -> x*x)) bs

