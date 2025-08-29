#!/usr/bin/env python3
"""
Compliance Checkers
==================

Target compliance checking functions.
"""

from typing import Dict, Any

from RAG.app.config import settings


def check_target_compliance(metrics: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
	"""
	Check if evaluation metrics meet target requirements.
	
	Args:
		metrics: Dictionary of evaluation metrics
		
	Returns:
		Dict with compliance status for each metric
	"""
	compliance = {}
	
	for metric_name, target_value in settings.evaluation.EVALUATION_TARGETS.items():
		actual_value = metrics.get(metric_name)
		
		if actual_value is not None:
			meets_target = actual_value >= target_value
			status = "✅ PASS" if meets_target else "❌ FAIL"
		else:
			meets_target = False
			status = "⚠️  NO DATA"
		
		compliance[metric_name] = {
			"target": target_value,
			"actual": actual_value,
			"meets_target": meets_target,
			"status": status
		}
	
	return compliance


def print_target_compliance(metrics: Dict[str, Any]):
	"""Print target compliance report."""
	compliance = check_target_compliance(metrics)
	
	print("\n📊 TARGET COMPLIANCE REPORT:")
	print("=" * 50)
	
	for metric_name, data in compliance.items():
		target = data["target"]
		actual = data["actual"]
		status = data["status"]
		
		if actual is not None:
			print(f"{metric_name:20} | Target: {target:.2f} | Actual: {actual:.3f} | {status}")
		else:
			print(f"{metric_name:20} | Target: {target:.2f} | Actual: N/A      | {status}")
	
	# Overall assessment
	passed = sum(data["meets_target"] for data in compliance.values())
	total = len(compliance)
	
	print("=" * 50)
	print(f"Overall: {passed}/{total} targets met")
	
	if passed == total:
		print("🎉 ALL TARGETS MET - EXCELLENT PERFORMANCE!")
	elif passed >= total * 0.8:
		print("🟡 MOST TARGETS MET - GOOD PERFORMANCE")
	else:
		print("🔴 MANY TARGETS MISSED - NEEDS IMPROVEMENT")
