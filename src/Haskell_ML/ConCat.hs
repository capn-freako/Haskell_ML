-- Interface to Conal's ConCat machinery for AD, etc.
--
-- Original author: David Banas <capn.freako@gmail.com>
-- Original date:   March 3, 2018
--
-- Copyright (c) 2018 David Banas; all rights reserved World wide.

{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}

{-|
Module      : Haskell_ML.ConCat
Description : Provides the Haskell_ML library with an interface to Conal Elliott's ConCat machinery.
Copyright   : (c) David Banas, 2018
License     : BSD-3
Maintainer  : capn.freako@gmail.com
Stability   : experimental
Portability : ?
-}
module Haskell_ML.ConCat where

import Prelude hiding ((.), unzip)

import           GHC.Generics (Par1(..),(:*:)(..),(:.:)(..))
import qualified Data.Vector.Sized as VS

import ConCat.Additive
import ConCat.Category
import ConCat.Deep
import ConCat.Misc
import ConCat.Orphans  (fstF, sndF)
import ConCat.Rebox    ()  -- Necessary for reboxing rules to fire
import ConCat.AltCat   ()  -- Necessary, but I've forgotten why.

import Haskell_ML.FCN (TrainEvo(..))
import Haskell_ML.Util

type V = VS.Vector

-- steps :: (C3 Summable p a b, Additive1 p, Functor f, Foldable f, Additive s, Num s)
--       => (p s -> a s -> b s) -> s -> f (a s :* b s) -> Unop (p s)

-- | Train a network on several epochs of the training data, keeping
-- track of accuracy and weight/bias changes per layer, after each.
trainNTimes :: (C3 Summable p a b, Additive1 p, Functor f, Foldable f, Additive s, Num s)
-- trainNTimes :: Int                                    -- ^ Number of epochs
            => Int                                    -- ^ Number of epochs
            -> s                                      -- ^ learning rate
            -> (p s -> a s -> b s)                    -- ^ The "network" just converts a set of parameters
                                                      -- into a function from input to output functors.
            -- -> ((V 10 --+ V 3) :*: (V 4 --+ V 10)) R
            -> p s                                    -- ^ initial guess for learnable parameters
            -- -> [(V 4 R, V 3 R)]                       -- ^ the training pairs
            -> f (a s :* b s)                         -- ^ the training pairs
            -- -> (((V 10 --+ V 3) :*: (V 4 --+ V 10)) R, TrainEvo)
            -> (p s, TrainEvo)
trainNTimes = trainNTimes' [] []

trainNTimes' :: (C3 Summable p a b, Additive1 p, Functor f, Foldable f, Additive s, Num s)
            => [Double]                    -- accuracies
            -> [([[Double]], [[Double]])]  -- weight/bias differences
            -> Int
            -> s
            -> (p s -> a s -> b s)
            -> p s
            -> f (a s :* b s)
            -> (p s, TrainEvo)
trainNTimes' accs diffs 0 _    _   ps _   = (ps, TrainEvo accs diffs)
trainNTimes' accs diffs n rate net ps prs = trainNTimes' (accs ++ [acc]) (diffs ++ [diff]) (n-1) rate net ps' prs
  where ps'        = steps net rate prs ps
        acc        = classificationAccuracy res ref
        (res, ref) = unzip $ fmap (toR . net ps' *** toR) prs
        diff       = ( zipWith (zipWith (-)) (getWeights ps') (getWeights ps)
                     , zipWith (zipWith (-)) (getBiases  ps') (getBiases  ps) )

-- getWeights :: ((V 10 --+ V 3) :*: (V 4 --+ V 10)) R -> [[Double]]
-- getWeights (w1 :*: w2) = foo w2 : [foo w1]
getWeights (w1 :*: w2) = getWeights w2 : [getWeights w1]
-- getWeights x@(a --+ b) = concat . map (VS.toList . fstF) . VS.toList $ unComp1 x
-- getWeights x@(Comp1 (g (f :*: Par1 b))) = concat . map (VS.toList . fstF) . VS.toList $ unComp1 x
getWeights x@(Comp1 _) = concat . map (VS.toList . fstF) . VS.toList $ unComp1 x
getWeights _           = error "Need more cases in Haskell_ML.ConCat.getWeights!"

getBiases :: ((V 10 --+ V 3) :*: (V 4 --+ V 10)) R -> [[Double]]
getBiases (w1 :*: w2) = getBiases w2 : [getBiases w1]
getBiases x@(Comp1 _) = map (unPar1 . sndF) . VS.toList $ unComp1 x
getBiases _           = error "Need more cases in Haskell_ML.ConCat.getBiases!"

