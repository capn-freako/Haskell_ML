-- General utilities for working with neural networks.
--
-- Original author: David Banas <capn.freako@gmail.com>
-- Original date:   January 22, 2018
--
-- Copyright (c) 2018 David Banas; all rights reserved World wide.

{-# OPTIONS_GHC -Wall #-}
{-# OPTIONS_GHC -Wno-unused-top-binds #-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeSynonymInstances #-}

{-|
Module      : Haskell_ML.Util
Description : Provides certain general purpose utilities in the Haskell_ML package.
Copyright   : (c) David Banas, 2018
License     : BSD-3
Maintainer  : capn.freako@gmail.com
Stability   : experimental
Portability : ?
-}
module Haskell_ML.Util where

import Prelude hiding (zipWith, zip)

import GHC.Generics (Par1(..),(:*:)(..),(:.:)(..))

import           Control.Monad.Trans.State.Lazy
import           Data.Finite
import           Data.Foldable
import           Data.Function                       (on)
import           Data.Key
import           Data.List                    hiding (zipWith, zip)
import           Data.Ord                            (comparing)
import           Data.Singletons.TypeLits
import qualified Data.Vector.Sized            as VS
import qualified Numeric.LinearAlgebra.Data   as LAD
import           Numeric.LinearAlgebra.Static hiding (mean)
import           System.Random
import           Text.Printf

import           ConCat.Deep
import           ConCat.Misc                         (sqr, result, cond)
import           ConCat.Orphans                      (fstF, sndF)

import           Haskell_ML.Classify.Classifiable


-- | Split a list of samples into training/testing sets.
--
-- The Finite given should be the percentage of samples desired for
-- training.
splitTrnTst :: Finite 101 -> [a] -> ([a],[a])
splitTrnTst n xs = splitAt (length xs * (fromInteger . getFinite) n `div` 100) xs


-- | Convert vector of Doubles from sized vector to hmatrix format.
toR :: KnownNat n => V n Double -> R n
toR = vector . VS.toList


-- | Calculate the classification accuracy, given:
--
--   - a functor of results vectors, and
--   - a functor of reference vectors.
classificationAccuracy :: (Foldable f, Keyed f, Eq (Key f), Foldable g, Zip g, Ord a)
                       => g (f a)  -- ^ Functor of result vectors.
                       -> g (f a)  -- ^ Functor of reference vectors.
                       -> Double
classificationAccuracy = (result.result) boolScore (zipWith ((==) `on` maxIndex))
 where boolScore = mean . fmap (cond 1 0)


-- | Mean value of a collection
mean :: (Foldable f, Fractional a) => f a -> a
mean = uncurry (/) . foldr (\e (s,c) -> (e+s,c+1)) (0,0)


-- | Prompt the user for input and read it.
prompt :: Read a => String -> IO a
prompt s = do
  putStrLn s
  readLn

-- | Pretty printer for values of type `R` n.
printVector :: (KnownNat n) => R n -> String
printVector v = foldl' (\ s x -> s ++ printf "%+6.4f  " x) "[ " ((LAD.toList . extract) v) ++ " ]"


-- | Pretty printer for values of type (`R` `m`, `R` `n`).
printVecPair :: (KnownNat m, KnownNat n) => (R m, R n) -> String
printVecPair (u, v) = "( " ++ printVector u ++ ", " ++ printVector v ++ " )"


-- | Show the learning evolution for the various parts/layers in a
-- list of HasLayers.
showPart :: (HasLayers p, RealFrac s, Ord s, PrintfArg s)
         => (p s -> [[s]])  -- ^ The "getter", which grabs the correct part of the parameters.
         -> String          -- ^ A label, which identifies the target of the getter.
         -> [p s]           -- ^ List of parameters, one per epoch.
         -> [String]
showPart f s ps =
  let xss = zip [1::Int,2..] $ (transpose . map f) ps
   in concatMap ( \ (i, xs) ->
                  [ "Average variance in layer " ++ show i ++ " " ++ s ++ ":"
                  , asciiPlot $ map (mean . map sqr) xs
                  ]
                ) xss


-- | Plot a list of Fractionals to an ASCII terminal.
asciiPlot :: (RealFrac a, Ord a, PrintfArg a) => [a] -> String
asciiPlot ys = unlines $
  zipWith (++)
    [ "        "
    , printf " %6.1e " x_max
    , "        "
    , "        "
    , "        "
    , "        "
    , "        "
    , "        "
    , "        "
    , "        "
    , "        "
    , printf " %6.1e " x_min
    , "        "
    ] $
    (:) "^" $ transpose (
    (:) "|||||||||||" $
    for xs $ \x ->
      valToStr $ (x - x_min) * 10 / x_range
    ) ++ ["|0" ++ concat [replicate 16 '_' ++ printf "%4d" (n * length ys `div` 3) | n <- [1..3]] ++ ">"]
      where valToStr  :: (RealFrac a, Ord a) => a -> String
            valToStr x = let i = round (10 - x)
                          in replicate i ' ' ++ "*" ++ replicate (10 - i) ' '
            x_min      = minimum xs
            x_max      = maximum xs
            x_range    = x_max - x_min
            xs         = takeEvery (length ys `div` 60) ys


-- | Return every Nth element of a list.
takeEvery :: Int -> [a] -> [a]
takeEvery n xs = case drop (n-1) xs of
                   (y:ys) -> y : takeEvery n ys
                   []     -> []

-- | Create an arbitrary functor filled with different random values.
randF :: (Traversable f, Applicative f, Random a) => Int -> f a
randF = evalState (sequenceA $ pure $ state random) . mkStdGen
{-# INLINE randF #-}


-- | Convenience function (= flip map).
for :: Functor f => f a -> (a -> b) -> f b
for = flip fmap


-- | Find the index of the maximum value in a keyed functor.
maxIndex :: (Foldable f, Keyed f, Ord a) => f a -> Key f
maxIndex = fst . maximumBy (comparing snd) . keyed


{--------------------------------------------------------------------
    Network structure
--------------------------------------------------------------------}

-- | A class of parameter types that can be split into a sequence of
-- layers of "weights" and "biases". The meaning of the two notions:
-- "weight" and "bias", are type specific. The only unifyng feature
-- of these, across types, is that they be fully resolved types.
class HasLayers p where
  getWeights :: p s -> [[s]]
  getBiases  :: p s -> [[s]]

instance (HasLayers f, HasLayers g) => HasLayers (g :*: f) where
  getWeights (g :*: f) = getWeights f ++ getWeights g
  getBiases  (g :*: f) = getBiases  f ++ getBiases  g

instance (Foldable a, Foldable b) => HasLayers (a --+ b) where
  getWeights (Comp1 gf) = [(concatMap (toList          . fstF) . toList) gf]
  getBiases  (Comp1 gf) = [(concatMap ((: []) . unPar1 . sndF) . toList) gf]

-- instance (HasLayers f, Foldable g) => HasLayers (g :.: f) where
--   getWeights (Comp1 g) = reverse $ foldl' (\ws -> (: ws) . concat . getWeights) [] g
--   getBiases  (Comp1 g) = reverse $ foldl' (\ws -> (: ws) . concat . getBiases)  [] g

-- instance Foldable f => HasLayers (f :*: Par1) where
--   getWeights (f :*: Par1 _) = [toList f]
--   getBiases  (_ :*: Par1 x) = [[x]]

-- instance Foldable f => HasLayers f where
--   getWeights f = [toList f]
--   getBiases  f = [[]]

