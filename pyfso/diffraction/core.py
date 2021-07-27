# -*- coding: utf-8 -*-

"""
Module with the abstract base classes for the definition of the composite
indicators as a processor pipeline.
"""
from __future__ import annotations
from abc import abstractclassmethod, ABC
from dataclasses import dataclass, field
import logging
from typing import List

from .dataloader import TradingDataset
from . import logger

logger.setLevel(logging.INFO)


class Processor(ABC):
    """Abstract base class for Processors classes
    

    .. versionadded:: 0.1
    """
    def __init__(self):
        logger.info(f"Initializing {self.__class__.__name__}")

    @abstractclassmethod
    def process(domain):
        """"""
        pass


class IndicatorABC(ABC):
    """Abstract base class for Composite Indicators
    
    
    .. versionadded:: 0.1
    """
    def __init__(self):
        print(f"Initializing {self.__class__.__name__}")

    @abstractclassmethod
    def build(domain):
        """Build the composite indicator"""
        pass

    @abstractclassmethod
    def export(domain):
        """Export results"""
        pass

    @abstractclassmethod
    def plot(domain):
        """Plot results"""
        pass


@dataclass
class CompositeIndicator(IndicatorABC):
    """Composite Indicator base class

    A technical indicator is a mathematical calculation based on historic price,
    volume, or (in the case of futures contracts) open interest information that
    aims to forecast financial market direction[1]. Nevertheless, trading
    strategies require multiple indicators and different combinations of them.
    Therefore, `techindicators` provides an interface to generate these
    "Composite Indicators" like a simple processor pipeline.


    Attributes
    ----------
    processors : List[Processor]
        Processing stages definend in a unique computational strcture.

    kwargs : dict
        Parameter dictionary for processors.


    References
    ----------
    .. [1] Wikipedia entry on the Technical indicator
        ..https://en.wikipedia.org/wiki/Technical_indicator


    .. versionadded:: 0.1    
    """
    processors: List[Processor]
    kwargs: field(default_factory=dict)

    def build(self, dataset: TradingDataset) -> TradingDataset:
        """Build the Composite Indicator
        
        Parameters
        ----------
        dataset : TradingDataset
            Data object that contains the financial time series dataframe
            (FTSF), interval, data source and symbol identifier.
        
        Returns
        -------
        dataset : TradingDataset
            Data object with the transformed financial time series dataframe
            (FTSF), interval, data source and symbol identifier.
        """
        for processor in self.processors:
            dataset = processor.process(dataset, **self.kwargs)

        self.dataset = dataset
        return self.dataset

    def export(self):
        """Export results"""
        pass

    def plot(self):
        """Plot results"""
        pass
