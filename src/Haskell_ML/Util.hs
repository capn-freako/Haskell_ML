-- General utilities for working with neural networks.
--
-- Original author: David Banas <capn.freako@gmail.com>
-- Original date:   January 22, 2018
--
-- Copyright (c) 2018 David Banas; all rights reserved World wide.

{-# OPTIONS_GHC -Wall #-}
{-# OPTIONS_GHC -Wno-unused-top-binds #-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{-|
Module      : Haskell_ML.Util
Description : Provides certain general purpose utilities in the Haskell_ML package.
Copyright   : (c) David Banas, 2018
License     : BSD-3
Maintainer  : capn.freako@gmail.com
Stability   : experimental
Portability : ?
-}
module Haskell_ML.Util
  ( Iris(..), Attributes(..), Sample
  , readIrisData, splitIrisData, splitTrnTst
  , attributeToVector, irisTypeToVector
  , classificationAccuracy, printVector, printVecPair, mkSmplsUniform
  , asciiPlot, calcMeanList
  , for, randF
  ) where

import           Control.Applicative
import           Control.Arrow
import           Control.Monad.Trans.State.Lazy
import           Data.List
import qualified Data.Text as T
import           Data.Attoparsec.Text hiding (take)
import           Data.Singletons.TypeLits
import           Numeric.LinearAlgebra.Data (maxIndex, toList)
import           Numeric.LinearAlgebra.Static
import           System.Random
import           Text.Printf

import           Haskell_ML.Classify.Classifiable


-- | Split a list of samples into training/testing sets.
--
-- The Finite given should be the percentage of samples desired for
-- training.
splitTrnTst :: Finite 101 -> [a] -> ([a],[a])
splitTrnTst _ [] = ([],[])
splitTrnTst n xs =
  let n'   = length xs * n `div` 100
      trn  = take n' xs
      tst  = drop n' xs
   in (trn, tst)


-- | Convert vector of Doubles from sized vector to hmatrix format.
toR :: V n a -> R n
toR = vector . VS.toList


-- | Convert vector of Doubles from hmatrix to sized vector format.
toV :: R n -> V n a
toV = VS.fromList . toList . unwrap


-- | Calculate the classification accuracy, given:
--
--   - a list of results vectors, and
--   - a list of reference vectors.
classificationAccuracy :: (KnownNat n) => [R n] -> [R n] -> Double
classificationAccuracy us vs = calcMeanList $ cmpr us vs

  where cmpr :: (KnownNat n) => [R n] -> [R n] -> [Double]
        cmpr xs ys = for (zipWith maxComp xs ys) $ \case
                       True  -> 1.0
                       False -> 0.0

        maxComp :: (KnownNat n) => R n -> R n -> Bool
        maxComp u v = maxIndex (extract u) == maxIndex (extract v)


-- | Calculate the mean value of a list.
calcMeanList :: (Fractional a) => [a] -> a
calcMeanList = uncurry (/) . foldr (\e (s,c) -> (e+s,c+1)) (0,0)


-- | Pretty printer for values of type `R` n.
printVector :: (KnownNat n) => R n -> String
printVector v = foldl' (\ s x -> s ++ printf "%+6.4f  " x) "[ " ((toList . extract) v) ++ " ]"


-- | Pretty printer for values of type (`R` `m`, `R` `n`).
printVecPair :: (KnownNat m, KnownNat n) => (R m, R n) -> String
printVecPair (u, v) = "( " ++ printVector u ++ ", " ++ printVector v ++ " )"


-- | Plot a list of Doubles to an ASCII terminal.
asciiPlot :: [Double] -> String
asciiPlot xs = unlines $
  zipWith (++)
    [ "        "
    , printf " %6.4f " x_max
    , "        "
    , "        "
    , "        "
    , "        "
    , "        "
    , "        "
    , "        "
    , "        "
    , "        "
    , printf " %6.4f " x_min
    , "        "
    ] $
    (:) "^" $ transpose (
    (:) "|||||||||||" $
    for (take 60 xs) $ \x ->
      valToStr $ (x - x_min) * 10 / x_range
    ) ++ ["|" ++ replicate 60 '_' ++ ">"]

      where valToStr  :: Double -> String
            valToStr x = let i = round (10 - x)
                          in replicate i ' ' ++ "*" ++ replicate (10 - i) ' '
            x_min      = minimum xs
            x_max      = maximum xs
            x_range    = x_max - x_min


-- | Create an arbitrary functor filled with different random values.
randF :: (Traversable f, Applicative f, Random a) => Int -> f a
randF = evalState (sequenceA $ pure $ state random) . mkStdGen
{-# INLINE randF #-}


-- | Convenience function (= flip map).
for :: [a] -> (a -> b) -> [b]
for = flip map

