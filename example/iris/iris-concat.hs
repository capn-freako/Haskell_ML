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

import           GHC.Generics (Par1(..),(:*:)(..),(:.:)(..))
import           Control.Arrow
import           Control.Monad
import           Data.Key     (Zip(..))
import           Data.List    hiding (zipWith, zip)
import           Data.Maybe   (fromMaybe)
import qualified Data.Vector.Sized as VS
import           System.Random.Shuffle

import ConCat.Deep
import ConCat.Misc     (R)
import ConCat.Orphans  (fstF, sndF)
import ConCat.Rebox    ()  -- Necessary for reboxing rules to fire
import ConCat.AltCat   ()  -- Necessary, but I've forgotten why.

import Haskell_ML.FCN  (TrainEvo(..))
import Haskell_ML.Util ( Sample, Attributes(..), Iris(..)
                       , readIrisData, mkSmplsUniform, splitTrnTst
                       , asciiPlot, calcMeanList
                       , randF, for
                       )

type V = VS.Vector

dataFileName :: String
dataFileName = "data/iris.csv"

main :: IO ()
main = do
  putStrLn "Learning rate?"
  rate <- readLn

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

  -- Create 2-layer network, using `ConCat.Deep`.
  let net = fmap (\x -> 2 * x - 1) $ randF 1
  putStrLn $ "Initial weights: " ++ show (getWeights net)
  putStrLn $ "Initial biases: " ++ show (getBiases net)

  -- let (n', TrainEvo{..}) = trainNTimes 60
  let (n', TrainEvo{..}) = trainNTimes 60
                                       rate
                                       net
                                       trnShuffled
      res = map (lr2 n' . fst) tstShuffled
      ref = map snd tstShuffled

  putStrLn $ "Final weights: " ++ show (getWeights n')
  putStrLn $ "Final biases: " ++ show (getBiases n')

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


-- | Train a network on several epochs of the training data, keeping
-- track of accuracy and weight/bias changes per layer, after each.
trainNTimes :: Int                                    -- ^ Number of epochs
            -> Double                                 -- ^ learning rate
            -> ((V 10 --+ V 3) :*: (V 4 --+ V 10)) R
            -> [(V 4 R, V 3 R)]                       -- ^ the training pairs
            -> (((V 10 --+ V 3) :*: (V 4 --+ V 10)) R, TrainEvo)
trainNTimes = trainNTimes' [] []

trainNTimes' :: [Double]                    -- accuracies
             -> [([[Double]], [[Double]])]  -- weight/bias differences
             -> Int
             -> Double
             -> ((V 10 --+ V 3) :*: (V 4 --+ V 10)) R
             -> [(V 4 R, V 3 R)]
             -> (((V 10 --+ V 3) :*: (V 4 --+ V 10)) R, TrainEvo)
trainNTimes' accs diffs 0 _    net _   = (net, TrainEvo accs diffs)
trainNTimes' accs diffs n rate net prs = trainNTimes' (accs ++ [acc]) (diffs ++ [diff]) (n-1) rate net' prs
  where net'  = steps lr2 rate prs net
  -- where net'  = zipWith (-) net $ steps lr2 rate prs net
        acc   = classificationAccuracy res ref
        res   = map (lr2 net' . fst) prs
        ref   = map snd prs
        diff  = ( zipWith (zipWith (-)) (getWeights net') (getWeights net)
                , zipWith (zipWith (-)) (getBiases  net') (getBiases  net) )

getWeights :: ((V 10 --+ V 3) :*: (V 4 --+ V 10)) R -> [[Double]]
getWeights (w1 :*: w2) = foo w2 : [foo w1]
  where foo = concat . map (VS.toList . fstF) . VS.toList . unComp1

getBiases :: ((V 10 --+ V 3) :*: (V 4 --+ V 10)) R -> [[Double]]
getBiases (w1 :*: w2) = bar w2 : [bar w1]
  where bar = map (unPar1 . sndF) . VS.toList . unComp1


-- | Split Iris dataset into classes and apply some preconditioning.
splitIrisData :: [Sample] -> ([(V 4 R, V 3 R)],[(V 4 R, V 3 R)],[(V 4 R, V 3 R)])
splitIrisData samps' =
  let samps1 = filter ((== Setosa)     . snd) samps'
      samps2 = filter ((== Versicolor) . snd) samps'
      samps3 = filter ((== Virginica)  . snd) samps'
      [samps1'', samps2'', samps3''] = (map . map) (attributeToVector *** irisTypeToVector) [samps1, samps2, samps3]
   in (samps1'', samps2'', samps3'')


-- | Convert a value of type `Attributes` to a value of type `V` 4.
attributeToVector :: Attributes -> V 4 R
attributeToVector Attributes{..} = fromMaybe (VS.replicate 0) $ VS.fromList [sepLen, sepWidth, pedLen, pedWidth]


-- | Convert a value of type `Iris` to a one-hot vector value of type `R` 3.
irisTypeToVector :: Iris -> V 3 R
irisTypeToVector = \case
  Setosa     -> fromMaybe (VS.replicate 0) $ VS.fromList [1,0,0]
  Versicolor -> fromMaybe (VS.replicate 0) $ VS.fromList [0,1,0]
  Virginica  -> fromMaybe (VS.replicate 0) $ VS.fromList [0,0,1]


-- | Calculate the classification accuracy, given:
--
--   - a list of results vectors, and
--   - a list of reference vectors.
classificationAccuracy :: [V 3 Double] -> [V 3 Double] -> Double
classificationAccuracy us vs = calcMeanList $ cmpr us vs

  where cmpr :: [V 3 Double] -> [V 3 Double] -> [Double]
        cmpr xs ys = for (zipWith maxComp xs ys) $ \case
                       True  -> 1.0
                       False -> 0.0

        maxComp :: V 3 Double -> V 3 Double -> Bool
        maxComp u v = VS.maxIndex u == VS.maxIndex v

