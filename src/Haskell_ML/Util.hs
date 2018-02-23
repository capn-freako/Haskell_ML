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
    f l = case parseOnly sampleParser l of
            Left msg -> error msg
            Right x  -> x


-- | Split Iris dataset into classes and apply some preconditioning.
splitIrisData :: [Sample] -> ([(R 4, R 3)],[(R 4, R 3)],[(R 4, R 3)])
splitIrisData samps' =
  let samps1 = filter ((== Setosa)     . snd) samps'
      samps2 = filter ((== Versicolor) . snd) samps'
      samps3 = filter ((== Virginica)  . snd) samps'
      [samps1'', samps2'', samps3''] = (map . map) (attributeToVector *** irisTypeToVector) [samps1, samps2, samps3]
   in (samps1'', samps2'', samps3'')


-- | Split a list of samples into training/testing sets.
splitTrnTst :: [a] -> ([a],[a])
splitTrnTst xs =
  let n   = length xs * 80 `div` 100
      trn = take n xs
      tst = drop n xs
   in (trn, tst)


-- | Rescale all feature values, to fall in [0,1].
mkSmplsUniform :: [Sample] -> [Sample]
mkSmplsUniform samps = map (first $ scaleAtt . offsetAtt) samps
  where scaleAtt :: Attributes -> Attributes
        scaleAtt Attributes{..} = Attributes (sls * sepLen) (sws * sepWidth) (pls * pedLen) (pws * pedWidth)

        offsetAtt :: Attributes -> Attributes
        offsetAtt Attributes{..} = Attributes (sepLen - slo) (sepWidth - swo) (pedLen - plo) (pedWidth - pwo)

        slo = minFldVal sepLen   samps
        swo = minFldVal sepWidth samps
        plo = minFldVal pedLen   samps
        pwo = minFldVal pedWidth samps

        sls = 1.0 / (maxFldVal sepLen   samps - slo)
        sws = 1.0 / (maxFldVal sepWidth samps - swo)
        pls = 1.0 / (maxFldVal pedLen   samps - plo)
        pws = 1.0 / (maxFldVal pedWidth samps - pwo)


-- | Finds the minimum value, for a particular `Attributes` field, in a
-- list of samples.
minFldVal :: (Attributes -> Double) -> [Sample] -> Double
minFldVal = overSamps minimum


-- | Finds the maximum value, for a particular `Attributes` field, in a
-- list of samples.
maxFldVal :: (Attributes -> Double) -> [Sample] -> Double
maxFldVal = overSamps maximum


-- | Applies a reduction to an `Attributes` field in a list of `Sample`s.
overSamps :: ([Double] -> Double) -> (Attributes -> Double) -> [Sample] -> Double
overSamps f fldAcc = f . fldFromSamps fldAcc


-- | Extracts the values of a `Attributes` field from a list of `Sample`s.
fldFromSamps :: (Attributes -> Double) -> [Sample] -> [Double]
fldFromSamps fldAcc = map (fldAcc . fst)


-- | Convert a value of type `Attributes` to a value of type `R` 4.
attributeToVector :: Attributes -> R 4
attributeToVector Attributes{..} = vector [sepLen, sepWidth, pedLen, pedWidth]


-- | Convert a value of type `Iris` to a one-hot vector value of type `R` 3.
irisTypeToVector :: Iris -> R 3
irisTypeToVector = \case
  Setosa     -> vector [1,0,0]
  Versicolor -> vector [0,1,0]
  Virginica  -> vector [0,0,1]


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


-- | Convenience function (= flip map).
for :: [a] -> (a -> b) -> [b]
for = flip map

