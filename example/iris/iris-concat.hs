-- Example use of `ConCat` to categorize the Iris dataset.
--
-- Original author: David Banas <capn.freako@gmail.com>
-- Original date:   February 21, 2018
--
-- Copyright (c) 2018 David Banas; all rights reserved World wide.
--
-- Note: this code started out as a direct copy of iris.hs.

{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators #-}

module Main where

import Prelude hiding (zipWith, zip)

import           GHC.Generics ((:*:)(..))
import           Control.Arrow
import           Control.Monad
import           Data.Foldable (fold)
import           Data.Key     (Zip(..))
import           Data.List    hiding (zipWith, zip)
import qualified Data.Vector.Sized as VS
import           System.Random.Shuffle

import ConCat.Deep
import ConCat.Misc     (R)
import ConCat.Rebox    ()  -- Necessary for reboxing rules to fire
import ConCat.AltCat   ()  -- Necessary, but I've forgotten why.

import Haskell_ML.Util
import Haskell_ML.Classify.Classifiable
import Haskell_ML.Classify.Iris

-- Define the network and its parameter type.
type PType = ((V 10 --+ V 3) :*: (V 4 --+ V 10)) R
net        = lr2  -- Defined in ConCat.Deep.

dataFileName :: String
dataFileName = "data/iris.csv"

main :: IO ()
main = do
  putStrLn "Learning rate?"
  rate <- readLn

  -- Read in the Iris data set. It contains an equal number of samples
  -- for all 3 classes of iris.
  putStrLn "Reading in data..."
  (samps :: [Sample Iris]) <- readClassifiableData dataFileName

  -- Shuffle samples.
  shuffled <- shuffleM samps

  -- Perform the following operations, in order:
  -- - Split samples according to class.
  -- - Within each class, make attribute values uniform over [0,1].
  -- - Split each class into training/testing sets.
  let splitV = VS.map ( splitTrnTst 80
                      . uncurry zip
                      . first mkAttrsUniform
                      . unzip
                      ) $ splitClassifiableData shuffled

  -- Gather up the training/testing sets into two lists and reshuffle.
  -- let (trn, tst) = VS.foldl mappend ([],[]) splitV
  -- let (trn, tst) = foldMap id splitV
  let (trn, tst) = fold splitV
  trnShuffled <- shuffleM trn
  tstShuffled <- shuffleM tst

  putStrLn "Done."

  -- Create 2-layer network, using `ConCat.Deep`.
  let ps         = (\x -> 2 * x - 1) <$> (randF 1 :: PType)
      ps'        = trainNTimes 60 rate net ps trnShuffled
      (res, ref) = unzip $ map (first (net (last ps'))) tstShuffled
      accs       = map (\p -> uncurry classificationAccuracy $ unzip $ map (first (net p)) trnShuffled) ps'
      diffs      = zipWith ((<*>) . fmap (-)) (tail ps') ps' :: [PType]

  putStrLn $ "Test accuracy: " ++ show (classificationAccuracy res ref)

  -- Plot the evolution of the training accuracy.
  putStrLn "Training accuracy:"
  putStrLn $ asciiPlot accs

  -- Plot the evolution of the weights and biases.
  let weights = zip [1::Int,2..] $ (transpose . map getWeights) diffs
      biases  = zip [1::Int,2..] $ (transpose . map getBiases)  diffs
  forM_ weights $ \ (i, ws) -> do
    putStrLn $ "Average variance in layer " ++ show i ++ " weights:"
    putStrLn $ asciiPlot $ map (calcMeanList . map (\x -> x*x)) ws
  forM_ biases $ \ (i, bs) -> do
    putStrLn $ "Average variance in layer " ++ show i ++ " biases:"
    putStrLn $ asciiPlot $ map (calcMeanList . map (\x -> x*x)) bs

