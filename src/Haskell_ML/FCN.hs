-- Building blocks for making fully connected neural networks (FCNs).
--
-- Original author: David Banas <capn.freako@gmail.com>
-- Original date:   January 18, 2018
--
-- Copyright (c) 2018 David Banas; all rights reserved World wide.

{-# LANGUAGE CPP #-}

-- Comment out, to use CCC for gradient solving.
-- #define USE_AD

{-# OPTIONS_GHC -Wall #-}
{-# OPTIONS_GHC -Wno-unused-top-binds #-}

{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ExplicitForAll #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}


{-|
Module      : Haskell_ML.FCN
Description : Allows: creation, training, running, saving, and loading,
              of multi-layer, fully connected neural networks.
Copyright   : (c) David Banas, 2018
                  Conal Elliott, 2018
License     : BSD-3
Maintainer  : capn.freako@gmail.com
Stability   : experimental
Portability : ?

This module is part of the @concat-learn@ package, which is maintained
in a @private@ GitHub repository. To request access, please, send
e-mail to: conal@conal.net
-}
module Haskell_ML.FCN
  ( FCNet,
    randNet, trainNet, runNet, netTest
  ) where

import Control.Monad.Random
import Data.Binary
import Data.List
import Data.Singletons.Prelude
import Data.Singletons.TypeLits
import GHC.Generics (Generic)
import Numeric.LinearAlgebra.Static


-- | A fully connected, multi-layer network with fixed input/output
-- widths, but variable (and existentially hidden!) internal structure.
--
-- Note: If you expose the inner value (of type `Network i hs o`), via
-- pattern matching, while `i` and `o` can be determined automatically,
-- via type inferencing, `hs` cannot, and you must be prepared to deal
-- with it in a completely polymorphic fashion. You may assume nothing
-- about `hs`, except that it is of kind `[Nat]`.
-- newtype (KnownNat i, KnownNat o) => FCNet i o = FCNet (Network i hs o)
data FCNet :: Nat -> Nat -> * where
  FCNet :: (Network i hs o) -> FCNet i o


-- | Returns a value of type `FCNet i o`, filled with random weights
-- ready for training, tucked inside the appropriate Monad, which must
-- be an instance of `MonadRandom`. (IO is such an instance.)
--
-- The input/output widths are determined by the compiler automatically,
-- via type inferencing.
--
-- The internal structure of the network is determined by the list of
-- integers passed in. Each integer in the list indicates the output
-- width of one hidden layer, with the first entry in the list
-- corresponding to the hidden layer nearest to the input layer.
randNet :: (KnownNat i, KnownNat o, MonadRandom m)
        => [Integer]
        -> m (FCNet i o)
randNet hs = withSomeSing hs $ \ss ->
  FCNet <$> randNetwork' ss


-- | Trains a value of type `FCNet i o`, using the supplied list of
-- training pairs (i.e. - matched input/output vectors).
trainNet :: (KnownNat i, KnownNat o)
         => Double        -- ^ learning rate
         -> FCNet i o     -- ^ the network to be trained
         -> [(R i, R o)]  -- ^ the training pairs
         -> FCNet i o     -- ^ the trained network
trainNet rate (FCNet net) trn_prs = FCNet $ sgd rate trn_prs net


-- | Run a network on a list of inputs.
runNet :: (KnownNat i, KnownNat o)
       => FCNet i o  -- ^ the network to run
       -> [R i]      -- ^ the list of inputs
       -> [R o]      -- ^ the list of outputs
runNet (FCNet n) in_prs = map (runNetwork n) in_prs


-- | `Binary` instance definition for `FCNet i o`.
--
-- With this definition, the user of our library is able to use standard
-- `put` and `get` calls, to serialize his created/trained network for
-- future use. And we don't need to provide auxilliary `saveNet` and
-- `loadNet` functions in the API.
instance (KnownNat i, KnownNat o) => Binary (FCNet i o) where
    put = putFCNet
    get = getFCNet


netTest :: MonadRandom m => Double -> Int -> m String
netTest rate n = do
    inps <- replicateM n $ do
      s <- getRandom
      return $ randomVector s Uniform * 2 - 1
    let outs = flip map inps $ \v ->
                 if v `inCircle` (fromRational 0.33, 0.33)
                      || v `inCircle` (fromRational (-0.33), 0.33)
                   then fromRational 1
                   else fromRational 0
    net0 :: Network 2 '[16, 8] 1 <- randNetwork
    -- let trained = foldl' trainEach net0 (zip inps outs)
    --       where
    --         trainEach :: (KnownNat i, SingI hs, KnownNat o)
    --                   => Network i hs o
    --                   -> (R i, R o)
    --                   -> Network i hs o
    --         trainEach nt (i, o) = trainNetwork rate i o nt
    let trained = sgd rate (zip inps outs) net0

        outMat = [ [ render (norm_2 (runNetwork trained (vector [x / 25 - 1,y / 10 - 1])))
                   | x <- [0..50] ]
                 | y <- [0..20] ]

        render r | r <= 0.2  = ' '
                 | r <= 0.4  = '.'
                 | r <= 0.6  = '-'
                 | r <= 0.8  = '='
                 | otherwise = '#'

    return $ unlines outMat
  where
    inCircle :: KnownNat n => R n -> (R n, Double) -> Bool
    v `inCircle` (o, r) = norm_2 (v - o) <= r


-----------------------------------------------------------------------
-- All following functions are for internal library use only!
-- They are not exported through the API.
-----------------------------------------------------------------------


-- A single network layer mapping an input of width `i` to an output of
-- width `o`, via simple matrix/vector mult.
data Layer i o = Layer { biases :: !(R o)
                       , nodes  :: !(L o i)
                       }
  deriving (Show, Generic)

instance (KnownNat i, KnownNat o) => Binary (Layer i o)


-- Generates a value of type `Layer i o`, filled with normally
-- distributed random values, tucked inside the appropriate Monad, which
-- must be an instance of `MonadRandom`.
randLayer :: (MonadRandom m, KnownNat i, KnownNat o)
          => m (Layer i o)
randLayer = do
  s1 :: Int <- getRandom
  s2 :: Int <- getRandom
  let b = randomVector  s1 Uniform * 2 - 1
      n = uniformSample s2 (-1) 1
  return $ Layer b n


-- General multi-layer network.
data Network :: Nat -> [Nat] -> Nat -> * where
  W    :: !(Layer i o)
       -> Network i '[] o
  (:&~) :: KnownNat h
       => !(Layer i h)
       -> !(Network h hs o)
       -> Network i (h ': hs) o
infixr 5 :&~


-- Returns a list of integers corresponding to the widths of the hidden
-- layers of a `Network i hs o`.
hiddenStruct :: Network i hs o -> [Integer]
hiddenStruct = \case
    W _    -> []
    _ :&~ (n' :: Network h hs' o)
           -> natVal (Proxy @h)
            : hiddenStruct n'


-- Generates a value of type `Network i hs o`
-- filled with random weights, ready to begin training.
--
-- Note: `hs` is determined explicitly, via the first argument, while
--       `i` and `o` are determined implicitly, via type inference.
randNetwork :: forall m i hs o. (MonadRandom m, KnownNat i, SingI hs, KnownNat o)
            => m (Network i hs o)
randNetwork = randNetwork' sing

randNetwork' :: forall m i hs o. (MonadRandom m, KnownNat i, KnownNat o)
             => Sing hs -> m (Network i hs o)
randNetwork' = \case
  SNil            -> W    <$> randLayer
  SNat `SCons` ss -> (:&~) <$> randLayer <*> randNetwork' ss


-- Binary instance definition for `Network i hs o`.
putNet :: (KnownNat i, KnownNat o)
       => Network i hs o
       -> Put
putNet = \case
    W w    -> put w
    w :&~ n -> put w *> putNet n

getNet :: forall i hs o. (KnownNat i, KnownNat o)
       => Sing hs
       -> Get (Network i hs o)
getNet = \case
    SNil            -> W    <$> get
    SNat `SCons` ss -> (:&~) <$> get <*> getNet ss

instance (KnownNat i, SingI hs, KnownNat o) => Binary (Network i hs o) where
    put = putNet
    get = getNet sing


putFCNet :: (KnownNat i, KnownNat o)
         => FCNet i o
         -> Put
putFCNet (FCNet net) = do
  put (hiddenStruct net)
  putNet net

getFCNet :: (KnownNat i, KnownNat o)
         => Get (FCNet i o)
getFCNet = do
  hs <- get
  withSomeSing hs $ \ss ->
    FCNet <$> getNet ss

runLayer :: (KnownNat i, KnownNat o)
         => Layer i o
         -> R i
         -> R o
runLayer (Layer b n) v = b + n #> v

runNetwork :: (KnownNat i, KnownNat o)
           => Network i hs o
           -> R i
           -> R o
runNetwork = \case
  W w        -> \(!v) -> logistic (runLayer w v)
  (w :&~ n') -> \(!v) -> let v' = logistic (runLayer w v)
                         in runNetwork n' v'

-- | Train a network of type `Network i hs o` using a list of training
-- pairs and the Stochastic Gradient Descent (SGD) approach.
sgd :: forall i hs o. (KnownNat i, KnownNat o)
    => Double           -- ^ learning rate
    -> [(R i, R o)]     -- ^ training pairs
    -> Network i hs o   -- ^ network to train
    -> Network i hs o   -- ^ trained network
sgd rate trn_prs net = foldl' (sgd_step rate) net trn_prs


-- | Train a network of type `Network i hs o` using a single training pair.
--
-- This code was taken directly from Justin Le's public GitHub archive:
-- https://github.com/mstksg/inCode/blob/43adae31b5689a95be83a72866600033fcf52b50/code-samples/dependent-haskell/NetworkTyped.hs#L77
-- and modified only slightly.
sgd_step :: forall i hs o. (KnownNat i, KnownNat o)
         => Double           -- ^ learning rate
         -> Network i hs o   -- ^ network to train
         -> (R i, R o)       -- ^ training pair
         -> Network i hs o   -- ^ trained network
sgd_step rate net trn_pr = fst $ go x0 net
  where
    x0     = fst trn_pr
    target = snd trn_pr
    go  :: forall j js. KnownNat j
        => R j              -- ^ input vector
        -> Network j js o   -- ^ network to train
        -> (Network j js o, R j)
    go !x (W w@(Layer wB wN))
        = let y    = runLayer w x
              o    = logistic y
              -- the gradient (how much y affects the error)
              --   (logistic' is the derivative of logistic)
              dEdy = logistic' y * (o - target)
              -- new bias weights and node weights
              wB'  = wB - konst rate * dEdy
              wN'  = wN - konst rate * (dEdy `outer` x)
              w'   = Layer wB' wN'
              -- bundle of derivatives for next step
              dWs  = tr wN #> dEdy
          in  (W w', dWs)
    -- handle the inner layers
    go !x (w@(Layer wB wN) :&~ n)
        = let y          = runLayer w x
              o          = logistic y
              -- get dWs', bundle of derivatives from rest of the net
              (n', dWs') = go o n
              -- the gradient (how much y affects the error)
              dEdy       = logistic' y * dWs'
              -- new bias weights and node weights
              wB'  = wB - konst rate * dEdy
              wN'  = wN - konst rate * (dEdy `outer` x)
              w'   = Layer wB' wN'
              -- bundle of derivatives for next step
              dWs  = tr wN #> dEdy
          in  (w' :&~ n', dWs)


-- Doesn't work, because the "constructors of R are not in scope."
-- What am I to do, here?!
-- Orphan `Ord` instance, for R n.
-- deriving instance (KnownNat n) => Ord (R n)


-- | Normalize a vector to a probability vector, via softmax.
-- softMax :: (KnownNat n)
--         => R n  -- ^ vector to be normalized
--         -> R n
-- softMax v = exp v / norm_0 v


-- Rectified Linear Unit
-- relu :: (KnownNat n)
--      => R n
--      -> R n
-- relu = max 0


-- relu' :: (KnownNat n)
--       => R n
--       -> R n
-- relu' v = if v > 0 then 1
--                    else 0


-- Logistic non-linear activation function.
logistic :: Floating a => a -> a
logistic x = 1 / (1 + exp (-x))

logistic' :: Floating a => a -> a
logistic' x = logix * (1 - logix)
  where
    logix = logistic x

