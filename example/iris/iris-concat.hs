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

-- import           GHC.Generics (Par1(..),(:*:)(..),(:.:)(..))
import           GHC.Generics ((:*:)(..))
import           Control.Arrow
import           Control.Monad
import           Data.Key     (Zip(..))
import           Data.List    hiding (zipWith, zip)
import qualified Data.Vector.Sized as VS
-- import           System.Random.Shuffle

import ConCat.Deep
import ConCat.Misc     (R)
-- import ConCat.Orphans  (fstF, sndF)
import ConCat.Rebox    ()  -- Necessary for reboxing rules to fire
import ConCat.AltCat   ()  -- Necessary, but I've forgotten why.

-- import Haskell_ML.FCN  (TrainEvo(..))
import Haskell_ML.Util
import Haskell_ML.Classify.Classifiable
import Haskell_ML.Classify.Iris

type PType = ((V 10 --+ V 3) :*: (V 4 --+ V 10)) R

dataFileName :: String
dataFileName = "data/iris.csv"

-- Parameter type definitions
-- newtype Affine2 a b c = Affine2 { unAffine2 :: (b --+ c) :*: (a --+ b) }

-- instance HasLayers Affine2 where
--   getWeights (Affine2 (g :*: f)) = [concat (getWeights t) | t <- (f, g)]
--   getBiases  (Affine2 (g :*: f)) = [concat (getBiases t)  | t <- (f, g)]

main :: IO ()
main = do
  putStrLn "Learning rate?"
  rate <- readLn

  -- Read in the Iris data set. It contains an equal number of samples
  -- for all 3 classes of iris.
  putStrLn "Reading in data..."
  (samps :: [Sample Iris]) <- readClassifiableData dataFileName

  -- Make field values uniform over [0,1] and split according to class,
  -- so we can keep equal representation throughout.
  -- let sampsV :: V 3 [(AttrVec Iris, TypeVec Iris)]
  --     sampsV = VS.map (uncurry zip . (first mkAttrsUniform) . unzip) $ splitClassifiableData samps

  -- Shuffle samples and split into training/testing groups.
  -- shuffled1 <- shuffleM samps1
  -- shuffled2 <- shuffleM samps2
  -- shuffled3 <- shuffleM samps3

  -- Split into training/testing groups.
  -- let splitV :: V 3 ([(AttrVec Iris, TypeVec Iris)],[(AttrVec Iris, TypeVec Iris)])
  --     splitV = VS.map (splitTrnTst 80) sampsV
  let splitV = VS.map ( splitTrnTst 80
                      . uncurry zip
                      . (first mkAttrsUniform)
                      . unzip
                      ) $ splitClassifiableData samps

  -- Reshuffle.
  -- trnShuffled <- shuffleM trn
  -- tstShuffled <- shuffleM tst

  let (trnShuffled, tstShuffled) = VS.foldl (uncurry (***) . ((++) *** (++))) ([],[]) splitV

  putStrLn "Done."

  -- Create 2-layer network, using `ConCat.Deep`.
  -- let ps  = fmap (\x -> 2 * x - 1) $ (randF 1 :: ((V 10 --+ V 3) :*: (V 4 --+ V 10)) R)
  let ps  = fmap (\x -> 2 * x - 1) $ (randF 1 :: PType)
      net = lr2
      -- (n', TrainEvo{..}) = trainNTimes 60
      ps' = trainNTimes 60 rate net ps trnShuffled
      (res, ref) = unzip $ map (first (net (last ps'))) tstShuffled
      accs = map (\p -> uncurry classificationAccuracy $ unzip $ map (first (net p)) trnShuffled) ps'
      -- diffs = zipWith (uncurry (***) . ((-) *** (-))) (tail ps') ps' :: [PType]
      diffs = zipWith ((<*>) . fmap (-)) (tail ps') ps' :: [PType]

  -- acc        = classificationAccuracy res ref
  -- -- (res, ref) = unzip $ fmap ((toR . net ps') *** toR) prs
  -- (res, ref) = unzip $ fmap (first (net ps')) prs
  -- diff       = ( zipWith (zipWith (-)) (getWeights ps') (getWeights ps)
  --              , zipWith (zipWith (-)) (getBiases  ps') (getBiases  ps) )

  putStrLn $ "Test accuracy: " ++ show (classificationAccuracy res ref)

  -- Plot the evolution of the training accuracy.
  putStrLn "Training accuracy:"
  putStrLn $ asciiPlot accs

  -- Plot the evolution of the weights and biases.
  -- let weights = zip [1::Int,2..] $ (transpose . map fst) diffs
  --     biases  = zip [1::Int,2..] $ (transpose . map snd) diffs
  let weights = zip [1::Int,2..] $ (transpose . map getWeights) diffs
      biases  = zip [1::Int,2..] $ (transpose . map getBiases)  diffs
  forM_ weights $ \ (i, ws) -> do
    putStrLn $ "Average variance in layer " ++ show i ++ " weights:"
    putStrLn $ asciiPlot $ map (calcMeanList . map (\x -> x*x)) ws
  forM_ biases $ \ (i, bs) -> do
    putStrLn $ "Average variance in layer " ++ show i ++ " biases:"
    putStrLn $ asciiPlot $ map (calcMeanList . map (\x -> x*x)) bs

-- -- | Data type for holding training evolution data.
-- data TrainEvo a = TrainEvo
--   { accs  :: [a]              -- ^ training accuracies
--   , diffs :: [([[a]],[[a]])]  -- ^ differences of weights/biases, by layer
--   }

-- instance Monoid (TrainEvo a) where
--   mempty      = TrainEvo [] []
--   mappend x y = TrainEvo (accs x ++ accs y) (diffs x ++ diffs y)

