from mmdet.evaluation import CocoMetric
from mmdet.registry import METRICS

@METRICS.register_module()
class FarmlandCocoMetric(CocoMetric):
    def compute_metrics(self, results):
        metrics = super().compute_metrics(results)
        
        # 클래스별 상세 분석 추가
        for i, class_name in enumerate(self.dataset_meta['classes']):
            class_ap = self.evaluate_single_class(results, i)
            metrics[f'{class_name}_AP'] = class_ap
            
        return metrics 