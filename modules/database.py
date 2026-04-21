"""
Database models for storing optimization history and user data
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import json

Base = declarative_base()

class User(Base):
    """User model for storing user information"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    models = relationship('ModelUpload', back_populates='user', cascade='all, delete-orphan')
    optimizations = relationship('OptimizationRun', back_populates='user', cascade='all, delete-orphan')

class ModelUpload(Base):
    """Store information about uploaded models"""
    __tablename__ = 'model_uploads'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    
    # Model information
    model_name = Column(String(255), nullable=False)
    model_type = Column(String(100))  # e.g., 'ResNet50', 'VGG16', 'Custom'
    upload_date = Column(DateTime, default=datetime.utcnow)
    
    # Original metrics
    original_size = Column(Float)  # MB
    original_latency = Column(Float)  # ms
    original_memory = Column(Float)  # MB
    original_accuracy = Column(Float)  # %
    total_parameters = Column(Integer)
    trainable_parameters = Column(Integer)
    
    # Additional metadata
    model_metadata = Column(JSON)  # Store additional info as JSON
    
    # Relationships
    user = relationship('User', back_populates='models')
    optimizations = relationship('OptimizationRun', back_populates='model', cascade='all, delete-orphan')

class OptimizationRun(Base):
    """Store optimization run results"""
    __tablename__ = 'optimization_runs'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    model_id = Column(Integer, ForeignKey('model_uploads.id'), nullable=False)
    
    # Run information
    run_date = Column(DateTime, default=datetime.utcnow)
    run_name = Column(String(255))  # Optional name for the run
    
    # Optimization techniques applied
    used_quantization = Column(Integer, default=0)  # Boolean as int
    used_pruning = Column(Integer, default=0)
    pruning_amount = Column(Float)  # 0.0 - 1.0
    used_distillation = Column(Integer, default=0)
    distillation_temperature = Column(Float)
    quantization_backend = Column(String(50))
    
    # Optimized metrics
    optimized_size = Column(Float)  # MB
    optimized_latency = Column(Float)  # ms
    optimized_memory = Column(Float)  # MB
    optimized_accuracy = Column(Float)  # %
    optimized_parameters = Column(Integer)
    
    # Improvements
    size_reduction_percent = Column(Float)
    latency_improvement_percent = Column(Float)
    memory_reduction_percent = Column(Float)
    accuracy_change_percent = Column(Float)
    
    # Carbon footprint
    carbon_saved_kg = Column(Float)
    energy_saved_kwh = Column(Float)
    
    # Additional data
    notes = Column(Text)  # User notes
    run_metadata = Column(JSON)  # Additional metadata
    
    # Relationships
    user = relationship('User', back_populates='optimizations')
    model = relationship('ModelUpload', back_populates='optimizations')

# Database manager class
class DatabaseManager:
    """Manager for database operations"""
    
    def __init__(self, db_path='sqlite:///optimization_history.db'):
        """Initialize database connection"""
        self.engine = create_engine(db_path, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def get_session(self):
        """Get a new database session"""
        return self.Session()
    
    def add_user(self, username, password_hash):
        """Add a new user"""
        session = self.get_session()
        try:
            user = User(username=username, password_hash=password_hash)
            session.add(user)
            session.commit()
            return user.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_user(self, username):
        """Get user by username"""
        session = self.get_session()
        try:
            user = session.query(User).filter_by(username=username).first()
            return user
        finally:
            session.close()
    
    def add_model_upload(self, user_id, model_name, model_type, metrics):
        """Add a new model upload"""
        session = self.get_session()
        try:
            model = ModelUpload(
                user_id=user_id,
                model_name=model_name,
                model_type=model_type,
                original_size=metrics.get('size'),
                original_latency=metrics.get('latency'),
                original_memory=metrics.get('memory'),
                original_accuracy=metrics.get('accuracy'),
                total_parameters=metrics.get('total_params'),
                trainable_parameters=metrics.get('trainable_params'),
                model_metadata=metrics  # Store full metrics as JSON
            )
            session.add(model)
            session.commit()
            return model.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def add_optimization_run(self, user_id, model_id, techniques, config, 
                            original_metrics, optimized_metrics, carbon_data=None):
        """Add a new optimization run"""
        session = self.get_session()
        try:
            # Calculate improvements
            size_reduction = ((original_metrics['size'] - optimized_metrics['size']) / 
                            original_metrics['size'] * 100)
            latency_improvement = ((original_metrics['latency'] - optimized_metrics['latency']) / 
                                  original_metrics['latency'] * 100)
            memory_reduction = ((original_metrics['memory'] - optimized_metrics['memory']) / 
                               original_metrics['memory'] * 100)
            accuracy_change = optimized_metrics['accuracy'] - original_metrics['accuracy']
            
            run = OptimizationRun(
                user_id=user_id,
                model_id=model_id,
                used_quantization=1 if techniques.get('quantization') else 0,
                used_pruning=1 if techniques.get('pruning') else 0,
                pruning_amount=config.get('prune_amount', 0) if techniques.get('pruning') else 0,
                used_distillation=1 if techniques.get('distillation') else 0,
                distillation_temperature=config.get('temperature', 0) if techniques.get('distillation') else 0,
                quantization_backend=config.get('quant_backend', ''),
                optimized_size=optimized_metrics['size'],
                optimized_latency=optimized_metrics['latency'],
                optimized_memory=optimized_metrics['memory'],
                optimized_accuracy=optimized_metrics['accuracy'],
                optimized_parameters=optimized_metrics['total_params'],
                size_reduction_percent=size_reduction,
                latency_improvement_percent=latency_improvement,
                memory_reduction_percent=memory_reduction,
                accuracy_change_percent=accuracy_change,
                carbon_saved_kg=carbon_data.get('carbon_kg', 0) if carbon_data else 0,
                energy_saved_kwh=carbon_data.get('energy_kwh', 0) if carbon_data else 0,
                run_metadata={
                    'original': original_metrics,
                    'optimized': optimized_metrics,
                    'techniques': techniques,
                    'config': config
                }
            )
            session.add(run)
            session.commit()
            return run.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_user_optimization_history(self, user_id, limit=50):
        """Get optimization history for a user"""
        session = self.get_session()
        try:
            runs = session.query(OptimizationRun).filter_by(user_id=user_id)\
                         .order_by(OptimizationRun.run_date.desc())\
                         .limit(limit).all()
            
            history = []
            for run in runs:
                techniques = []
                if run.used_quantization:
                    techniques.append('Quantization')
                if run.used_pruning:
                    techniques.append(f'Pruning ({run.pruning_amount*100:.0f}%)')
                if run.used_distillation:
                    techniques.append('Distillation')
                
                history.append({
                    'id': run.id,
                    'timestamp': run.run_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'model_name': run.model.model_name if run.model else 'Unknown',
                    'techniques': ', '.join(techniques),
                    'original_size': run.model.original_size if run.model else 0,
                    'model_size': run.optimized_size,
                    'size_reduction': f"{run.size_reduction_percent:.1f}%",
                    'accuracy': f"{run.optimized_accuracy:.2f}%",
                    'latency': f"{run.optimized_latency:.2f}ms",
                    'carbon_saved': f"{run.carbon_saved_kg:.2f} kg",
                })
            
            return history
        finally:
            session.close()
    
    def get_optimization_run(self, run_id):
        """Get detailed information about a specific run"""
        session = self.get_session()
        try:
            run = session.query(OptimizationRun).filter_by(id=run_id).first()
            if run:
                return {
                    'id': run.id,
                    'date': run.run_date,
                    'model_name': run.model.model_name if run.model else 'Unknown',
                    'techniques': {
                        'quantization': bool(run.used_quantization),
                        'pruning': bool(run.used_pruning),
                        'distillation': bool(run.used_distillation)
                    },
                    'original_metrics': run.run_metadata.get('original') if run.run_metadata else {},
                    'optimized_metrics': run.run_metadata.get('optimized') if run.run_metadata else {},
                    'improvements': {
                        'size_reduction': run.size_reduction_percent,
                        'latency_improvement': run.latency_improvement_percent,
                        'memory_reduction': run.memory_reduction_percent,
                        'accuracy_change': run.accuracy_change_percent
                    },
                    'carbon': {
                        'saved_kg': run.carbon_saved_kg,
                        'saved_kwh': run.energy_saved_kwh
                    }
                }
            return None
        finally:
            session.close()
    
    def get_user_stats(self, user_id):
        """Get aggregate statistics for a user"""
        session = self.get_session()
        try:
            runs = session.query(OptimizationRun).filter_by(user_id=user_id).all()
            models = session.query(ModelUpload).filter_by(user_id=user_id).all()
            
            if not runs:
                return None
            
            total_size_saved = sum((run.model.original_size - run.optimized_size) 
                                  for run in runs if run.model)
            total_carbon_saved = sum(run.carbon_saved_kg for run in runs)
            total_energy_saved = sum(run.energy_saved_kwh for run in runs)
            avg_size_reduction = sum(run.size_reduction_percent for run in runs) / len(runs)
            
            return {
                'total_optimizations': len(runs),
                'total_models': len(models),
                'total_size_saved_mb': total_size_saved,
                'total_carbon_saved_kg': total_carbon_saved,
                'total_energy_saved_kwh': total_energy_saved,
                'avg_size_reduction_percent': avg_size_reduction,
                'most_recent_run': runs[0].run_date if runs else None
            }
        finally:
            session.close()
    
    def delete_optimization_run(self, run_id, user_id):
        """Delete an optimization run"""
        session = self.get_session()
        try:
            run = session.query(OptimizationRun).filter_by(id=run_id, user_id=user_id).first()
            if run:
                session.delete(run)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

# Initialize global database manager
_db_manager = None

def get_db_manager():
    """Get or create the global database manager"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
