-- Class of types amenable to classification, via machine learning.
--
-- Original author: David Banas <capn.freako@gmail.com>
-- Original date:   March 1, 2018
--
-- Copyright (c) 2018 David Banas; all rights reserved World wide.
--
-- Note: This code originally came into being, as part of a code clean-
--       up, which started at 'v0.5.0' of the Haskell_ML repository.

{-# OPTIONS_GHC -Wall #-}
{-# OPTIONS_GHC -Wno-unused-top-binds #-}

{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

{-|
Module      : Haskell_ML.Classify.Classifiable
Description : A class of types amenable to classification, via machine learning.
Copyright   : (c) David Banas, 2018
License     : BSD-3
Maintainer  : capn.freako@gmail.com
Stability   : experimental
Portability : ?
-}
module Haskell_ML.Classify.Classifiable where

import           GHC.TypeLits
import           Control.Arrow               ((***), (&&&))
import           Data.Attoparsec.Text hiding (take)
import           Data.Foldable               (foldl')
import qualified Data.Text         as T
import qualified Data.Vector.Sized as VS

type V = VS.Vector


-- | Class of types amenable to classification, via machine learning.
class Classifiable t where
  type Card t :: Nat
  type NAtt t :: Nat
  data Attr t :: *
  type Sample  t :: *
  type Sample  t = (Attr t, t)
  type AttrVec t :: *
  type AttrVec t = V (NAtt t) Double
  type TypeVec t :: *
  type TypeVec t = V (Card t) Double
  filtPreds    :: V (Card t) (t -> Bool)
  sampleParser :: Parser (Attr t, t)
  attrToVec    :: Attr t -> V (NAtt t) Double
  typeToVec    :: t      -> TypeVec t


-- | Read a list of samples from the named file.
--
-- Note: Can throw error!
readClassifiableData :: Classifiable t => String -> IO [(Attr t, t)]
readClassifiableData fname = do
    ls <- T.lines . T.pack <$> readFile fname
    return $ f <$> ls
  where
    f l = case parseOnly sampleParser l of
            Left msg -> error msg
            Right x  -> x


-- | Split a list of samples into classes and convert to vector form.
splitClassifiableData :: (Classifiable t, KnownNat (NAtt t)) => [(Attr t, t)] -> V (Card t) [(V (NAtt t) Double, TypeVec t)]
splitClassifiableData samps =
  VS.map (\q -> map (attrToVec *** typeToVec) $ filter (q . snd) samps) filtPreds


-- | Rescale all attribute values to fall in [0,1].
mkAttrsUniform :: KnownNat n => [V n Double] -> [V n Double]
mkAttrsUniform [] = []
mkAttrsUniform vs = map (VS.zipWith (*) scales . flip (VS.zipWith (-)) mins) vs
  where (mins, maxs) = getAttrRanges vs
        scales       = VS.zipWith (\ mn mx -> if mx == mn then 0 else 1 / (mx - mn)) mins maxs

-- | Find the extremes in a list of attribute vectors.
getAttrRanges :: KnownNat n => [V n Double] -> (V n Double, V n Double)
getAttrRanges = f min 1e10 &&& f max (-1e10)
  where f g x = foldl' (VS.zipWith g) (VS.replicate x)

