from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    PageBreak, Image
)
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
import tempfile
from pathlib import Path

def generate_optimization_report(original_metrics, optimized_metrics, optimization_config, output_path):
    """
    Generate a comprehensive PDF report of the optimization process
    """
    # Create document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    # Container for story
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2C3E50'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#34495E'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    story.append(Paragraph("AI Model Optimization Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Report metadata
    report_date = datetime.now().strftime('%B %d, %Y at %H:%M')
    story.append(Paragraph(f"<b>Generated:</b> {report_date}", styles['Normal']))
    story.append(Paragraph(f"<b>Report Type:</b> Model Optimization Analysis", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    
    size_reduction = ((original_metrics['size'] - optimized_metrics['size']) / original_metrics['size']) * 100
    latency_improvement = ((original_metrics['latency'] - optimized_metrics['latency']) / original_metrics['latency']) * 100
    accuracy_change = optimized_metrics['accuracy'] - original_metrics['accuracy']
    
    summary_text = f"""
    This report presents the results of applying advanced optimization techniques to a machine learning model.
    The optimization process achieved a <b>{size_reduction:.1f}%</b> reduction in model size and 
    <b>{latency_improvement:.1f}%</b> improvement in inference latency, while maintaining accuracy within 
    <b>{abs(accuracy_change):.2f}%</b> of the original model.
    """
    
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Optimization Techniques Applied
    story.append(Paragraph("Optimization Techniques Applied", heading_style))
    
    techniques_list = optimization_config.get('techniques', [])
    if techniques_list:
        for technique in techniques_list:
            story.append(Paragraph(f"• {technique}", styles['Normal']))
    else:
        story.append(Paragraph("• Quantization", styles['Normal']))
        story.append(Paragraph("• Pruning", styles['Normal']))
    
    story.append(Spacer(1, 0.3*inch))
    
    # Performance Metrics Comparison
    story.append(Paragraph("Performance Metrics Comparison", heading_style))
    
    # Create comparison table
    comparison_data = [
        ['Metric', 'Original Model', 'Optimized Model', 'Improvement'],
        ['Model Size (MB)', f"{original_metrics['size']:.2f}", f"{optimized_metrics['size']:.2f}", f"{size_reduction:.1f}%"],
        ['Latency (ms)', f"{original_metrics['latency']:.2f}", f"{optimized_metrics['latency']:.2f}", f"{latency_improvement:.1f}%"],
        ['Memory Usage (MB)', f"{original_metrics['memory']:.2f}", f"{optimized_metrics['memory']:.2f}", 
         f"{((original_metrics['memory'] - optimized_metrics['memory']) / original_metrics['memory'] * 100):.1f}%"],
        ['Accuracy (%)', f"{original_metrics['accuracy']:.2f}", f"{optimized_metrics['accuracy']:.2f}", f"{accuracy_change:+.2f}%"],
        ['Total Parameters', f"{original_metrics['total_params']:,}", f"{optimized_metrics['total_params']:,}", 
         f"{((original_metrics['total_params'] - optimized_metrics['total_params']) / original_metrics['total_params'] * 100):.1f}%"]
    ]
    
    table = Table(comparison_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    
    story.append(table)
    story.append(Spacer(1, 0.4*inch))
    
    # Key Findings
    story.append(Paragraph("Key Findings", heading_style))
    
    findings = [
        f"<b>Model Size:</b> Reduced from {original_metrics['size']:.2f} MB to {optimized_metrics['size']:.2f} MB",
        f"<b>Inference Speed:</b> Improved by {latency_improvement:.1f}%, resulting in faster predictions",
        f"<b>Accuracy:</b> {('Maintained' if abs(accuracy_change) < 1 else 'Slightly decreased')} with {accuracy_change:+.2f}% change",
        f"<b>Memory Efficiency:</b> Reduced memory footprint by {((original_metrics['memory'] - optimized_metrics['memory']) / original_metrics['memory'] * 100):.1f}%",
        f"<b>Deployment Ready:</b> Optimized model is suitable for {'edge and mobile' if optimized_metrics['size'] < 50 else 'cloud'} deployment"
    ]
    
    for finding in findings:
        story.append(Paragraph(f"• {finding}", styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
    
    story.append(Spacer(1, 0.3*inch))
    
    # Recommendations
    story.append(Paragraph("Recommendations", heading_style))
    
    recommendations_text = """
    Based on the optimization results, we recommend the following:
    <br/><br/>
    <b>1. Deployment Strategy:</b> The optimized model is well-suited for production deployment. 
    Consider exporting to ONNX format for cross-platform compatibility.
    <br/><br/>
    <b>2. Further Testing:</b> Conduct thorough testing with real-world data to validate the 
    accuracy metrics and ensure the model performs as expected in production scenarios.
    <br/><br/>
    <b>3. Monitoring:</b> Implement continuous monitoring of model performance, latency, and 
    accuracy in production to detect any degradation over time.
    <br/><br/>
    <b>4. Future Optimization:</b> Consider additional techniques such as knowledge distillation 
    or neural architecture search for even greater compression if needed.
    """
    
    story.append(Paragraph(recommendations_text, styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Page break for next section
    story.append(PageBreak())
    
    # Technical Details
    story.append(Paragraph("Technical Details", heading_style))
    
    # Configuration details
    story.append(Paragraph("<b>Optimization Configuration:</b>", styles['Normal']))
    story.append(Spacer(1, 0.1*inch))
    
    if 'prune_amount' in optimization_config:
        story.append(Paragraph(f"• Pruning Amount: {optimization_config['prune_amount']*100:.0f}%", styles['Normal']))
    
    if 'quant_backend' in optimization_config:
        story.append(Paragraph(f"• Quantization Backend: {optimization_config['quant_backend']}", styles['Normal']))
    
    if 'temperature' in optimization_config:
        story.append(Paragraph(f"• Distillation Temperature: {optimization_config['temperature']}", styles['Normal']))
    
    story.append(Spacer(1, 0.3*inch))
    
    # Model Architecture Summary
    story.append(Paragraph("<b>Model Architecture Summary:</b>", styles['Normal']))
    story.append(Spacer(1, 0.1*inch))
    
    arch_data = [
        ['Component', 'Original', 'Optimized'],
        ['Total Parameters', f"{original_metrics['total_params']:,}", f"{optimized_metrics['total_params']:,}"],
        ['Trainable Parameters', f"{original_metrics['trainable_params']:,}", f"{optimized_metrics['trainable_params']:,}"],
        ['Model Size (MB)', f"{original_metrics['size']:.2f}", f"{optimized_metrics['size']:.2f}"]
    ]
    
    arch_table = Table(arch_data, colWidths=[2.5*inch, 2*inch, 2*inch])
    arch_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ECC71')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10)
    ]))
    
    story.append(arch_table)
    story.append(Spacer(1, 0.4*inch))
    
    # Conclusion
    story.append(Paragraph("Conclusion", heading_style))
    
    conclusion_text = f"""
    The optimization process successfully reduced the model size by <b>{size_reduction:.1f}%</b> 
    while maintaining competitive accuracy. The optimized model demonstrates improved inference 
    speed and reduced memory footprint, making it suitable for deployment in resource-constrained 
    environments. These improvements translate to lower operational costs, reduced energy consumption, 
    and better user experience through faster response times.
    <br/><br/>
    The model is now ready for export and deployment to production environments.
    """
    
    story.append(Paragraph(conclusion_text, styles['Normal']))
    story.append(Spacer(1, 0.5*inch))
    
    # Footer
    footer_text = "Generated by AI Model Optimization Platform"
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.grey,
        alignment=TA_CENTER
    )
    story.append(Paragraph(footer_text, footer_style))
    
    # Build PDF
    doc.build(story)
    
    return output_path
