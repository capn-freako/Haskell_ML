-- Types and utilities for performing classification on the iris database.
--
-- Original author: David Banas <capn.freako@gmail.com>
-- Original date:   March 1, 2018
--
-- Copyright (c) 2018 David Banas; all rights reserved World wide.
--
-- Note: This code was originally culled from Util.hs.

{-# LANGUAGE LambdaCase #-}

{-# OPTIONS_GHC -Wall #-}
{-# OPTIONS_GHC -Wno-unused-top-binds #-}

{-|
Module      : Haskell_ML.Classify.Iris
Description : Provides certain utilities for working with the Iris database.
Copyright   : (c) David Banas, 2018
License     : BSD-3
Maintainer  : capn.freako@gmail.com
Stability   : experimental
Portability : ?
-}
module Haskell_ML.Classify.Iris
  ( Iris(..), IrisAttr(..)
  ) where

import Haskell_ML.Classify.Classifiable


-- | The 3 classes of iris are represented by the 3 constructors of this
-- type.
data Iris = Setosa
          | Versicolor
          | Virginica
  deriving (Show, Read, Eq, Ord, Enum)


-- | Data type representing the set of attributes for a sample in the
-- Iris dataset.
data IrisAttr = IrisAttr
  { sepLen   :: Double
  , sepWidth :: Double
  , pedLen   :: Double
  , pedWidth :: Double
  } deriving (Show, Read, Eq, Ord)


instance Classifiable Iris where
  type Card Iris = 3
  type NAtt Iris = 4
  type Attr Iris = IrisAttr
  attrToVec IrisAttr{..} = VS.fromList [sepLen, sepWidth, pedLen, pedWidth]
  typeToVec = \case
    Setosa     -> VS.fromList [1,0,0]
    Versicolor -> VS.fromList [0,1,0]
    Virginica  -> VS.fromList [0,0,1]
  filtPreds = VS.fromList [ (== Setosa)
                          , (== Versicolor)
                          , (== Virginica)
                          ]
  sampleParser = f <$> (double <* char ',')
                   <*> (double <* char ',')
                   <*> (double <* char ',')
                   <*> (double <* char ',')
                   <*> irisParser
    where
      f sl sw pl pw i = (Attributes sl sw pl pw, i)
      irisParser :: Parser Iris
      irisParser =     string "Iris-setosa"     *> return Setosa
                   <|> string "Iris-versicolor" *> return Versicolor
                   <|> string "Iris-virginica"  *> return Virginica

