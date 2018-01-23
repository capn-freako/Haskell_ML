-- General utilities for working with neural networks.
--
-- Original author: David Banas <capn.freako@gmail.com>
-- Original date:   January 22, 2018
--
-- Copyright (c) 2018 David Banas; all rights reserved World wide.

{-# OPTIONS_GHC -Wall #-}
{-# OPTIONS_GHC -Wno-unused-top-binds #-}

{-# LANGUAGE OverloadedStrings #-}

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
  ( Iris(..), Attributes, Sample
  , readIrisData
  ) where

import           Control.Applicative
import qualified Data.Text as T
import           Data.Attoparsec.Text


-- | The 3 classes of iris are represented by the 3 constructors of this
-- type.
data Iris = Setosa
          | Versicolor
          | Virginica
  deriving (Show, Read, Eq, Ord, Enum)


-- | Data type representing the set of attributes for a sample in the
-- Iris dataset.
data Attributes = Attributes
  { sepLen   :: Double
  , sepWidth :: Double
  , pedLen   :: Double
  , pedWidth :: Double
  } deriving (Show, Read, Eq, Ord)


-- | A single sample in the dataset is a pair of a list of attributes
-- and a classification.
type Sample = (Attributes, Iris)


-- | Read in an Iris dataset from the given file name.
readIrisData :: String -> IO [Sample]
readIrisData fname = do
    ls <- T.lines . T.pack <$> readFile fname
    return $ f <$> ls

  where
    -- f l = let Right x = parseOnly sampleParser l
    --        in x
    f l = case (parseOnly sampleParser l) of
            Left msg -> error msg
            Right x  -> x


-----------------------------------------------------------------------
-- All following functions are for internal library use only!
-- They are not exported through the API.
-----------------------------------------------------------------------


sampleParser :: Parser Sample
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

