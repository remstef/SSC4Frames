from factory_manager import FactoryManager

from ssc4frames.clusterer import DBClusterer, MultiClusterer, OrderedMultiClusterer


def create_factory():

  factory = FactoryManager()
  factory.add_object_hierarchy('dbclusterer', DBClusterer)
  factory.add_object_hierarchy('multiclusterer', MultiClusterer)
  factory.add_object_hierarchy('orderedmulticlusterer', OrderedMultiClusterer)

  return factory

