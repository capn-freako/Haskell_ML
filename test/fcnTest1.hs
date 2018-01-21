-- Test of FCN module
--
-- Original author: David Banas <capn.freako@gmail.com>
-- Original date:   January 20, 2018
--
-- Copyright (c) 2018 David Banas; all rights reserved World wide.

module Main where

import Control.Monad.Random
import Data.Maybe
import System.Environment
import Text.Read
import Haskell_ML.FCN

main :: IO ()
main = do
    args <- getArgs
    let n    = readMaybe =<< (args !!? 0)
        rate = readMaybe =<< (args !!? 1)
    putStrLn "Training network..."
    putStrLn =<< evalRandIO (netTest (fromMaybe 0.25   rate)
                                     (fromMaybe 500000 n   )
                            )

(!!?) :: [a] -> Int -> Maybe a
xs !!? i = listToMaybe (drop i xs)

