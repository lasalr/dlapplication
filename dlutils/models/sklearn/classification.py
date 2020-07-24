    def getSKLearnLogisticRegression(self, regParam, dim=1):
        from DLplatform.learning.batch.sklearnClassifiers import LogisticRegression
        
        learner = LogisticRegression(regParam = regParam, dim = dim)
        return learner

    def get_sklearn_sv_classification(self, regParam, dim):
            # TODO Implement similar function to getSKLearnLogisticRegression()
            pass