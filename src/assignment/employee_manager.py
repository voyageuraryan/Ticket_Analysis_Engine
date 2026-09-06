"""
Employee management for ticket assignment.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from loguru import logger


class EmployeeManager:
    """Manage employee data and availability."""
    
    def __init__(self, data_dir: Path):
        """
        Initialize employee manager.
        
        Args:
            data_dir: Directory to store employee data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.master_file = self.data_dir / "employees_master.json"
        self.daily_file = self.data_dir / "daily_assignments.json"
        
        self._load_data()
    
    def _load_data(self):
        """Load employee data from files."""
        # Load master data
        if self.master_file.exists():
            with open(self.master_file, 'r') as f:
                self.master_data = json.load(f)
        else:
            self.master_data = {
                'employees': [],
                'last_updated': None
            }
        
        # Load daily data
        if self.daily_file.exists():
            with open(self.daily_file, 'r') as f:
                self.daily_data = json.load(f)
        else:
            self.daily_data = {
                'date': None,
                'assignments': {}
            }
        
        # Reset daily data if it's a new day
        self._check_daily_reset()
    
    def _check_daily_reset(self):
        """Reset daily assignments if it's a new day."""
        today = datetime.now().strftime('%Y-%m-%d')
        
        if self.daily_data['date'] != today:
            logger.info(f"New day detected. Resetting daily assignments.")
            self.daily_data = {
                'date': today,
                'assignments': {emp['id']: 0 for emp in self.master_data['employees']}
            }
            self._save_daily_data()
    
    def _save_master_data(self):
        """Save master employee data."""
        self.master_data['last_updated'] = datetime.now().isoformat()
        with open(self.master_file, 'w') as f:
            json.dump(self.master_data, f, indent=2)
        logger.info(f"Master data saved: {len(self.master_data['employees'])} employees")
    
    def _save_daily_data(self):
        """Save daily assignment data."""
        with open(self.daily_file, 'w') as f:
            json.dump(self.daily_data, f, indent=2)
    
    def add_employee(
        self,
        employee_id: str,
        name: str,
        modules: List[str],
        email: Optional[str] = None,
        max_daily_tickets: int = 20
    ) -> Dict:
        """
        Add a new employee.
        
        Args:
            employee_id: Unique employee ID
            name: Employee name
            modules: List of SAP modules the employee handles
            email: Employee email (optional)
            max_daily_tickets: Maximum tickets per day
            
        Returns:
            Employee data dictionary
        """
        # Check if employee already exists
        existing = self.get_employee(employee_id)
        if existing:
            raise ValueError(f"Employee {employee_id} already exists")
        
        employee = {
            'id': employee_id,
            'name': name,
            'modules': modules,
            'email': email,
            'max_daily_tickets': max_daily_tickets,
            'status': 'available',  # available, on_leave, busy
            'total_assigned': 0,
            'created_at': datetime.now().isoformat()
        }
        
        self.master_data['employees'].append(employee)
        self.daily_data['assignments'][employee_id] = 0
        
        self._save_master_data()
        self._save_daily_data()
        
        logger.info(f"Added employee: {name} ({employee_id}) - Modules: {modules}")
        return employee
    
    def update_employee(
        self,
        employee_id: str,
        name: Optional[str] = None,
        modules: Optional[List[str]] = None,
        email: Optional[str] = None,
        max_daily_tickets: Optional[int] = None,
        status: Optional[str] = None
    ) -> Dict:
        """
        Update employee information.
        
        Args:
            employee_id: Employee ID to update
            name: New name (optional)
            modules: New modules list (optional)
            email: New email (optional)
            max_daily_tickets: New max tickets (optional)
            status: New status (optional)
            
        Returns:
            Updated employee data
        """
        employee = self.get_employee(employee_id)
        if not employee:
            raise ValueError(f"Employee {employee_id} not found")
        
        # Update fields
        if name is not None:
            employee['name'] = name
        if modules is not None:
            employee['modules'] = modules
        if email is not None:
            employee['email'] = email
        if max_daily_tickets is not None:
            employee['max_daily_tickets'] = max_daily_tickets
        if status is not None:
            if status not in ['available', 'on_leave', 'busy']:
                raise ValueError(f"Invalid status: {status}")
            employee['status'] = status
        
        employee['updated_at'] = datetime.now().isoformat()
        
        self._save_master_data()
        
        logger.info(f"Updated employee: {employee['name']} ({employee_id})")
        return employee
    
    def delete_employee(self, employee_id: str) -> bool:
        """
        Delete an employee.
        
        Args:
            employee_id: Employee ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        initial_count = len(self.master_data['employees'])
        self.master_data['employees'] = [
            emp for emp in self.master_data['employees']
            if emp['id'] != employee_id
        ]
        
        if len(self.master_data['employees']) < initial_count:
            # Remove from daily assignments
            if employee_id in self.daily_data['assignments']:
                del self.daily_data['assignments'][employee_id]
            
            self._save_master_data()
            self._save_daily_data()
            
            logger.info(f"Deleted employee: {employee_id}")
            return True
        
        return False
    
    def get_employee(self, employee_id: str) -> Optional[Dict]:
        """
        Get employee by ID.
        
        Args:
            employee_id: Employee ID
            
        Returns:
            Employee data or None
        """
        for emp in self.master_data['employees']:
            if emp['id'] == employee_id:
                return emp
        return None
    
    def get_all_employees(self) -> List[Dict]:
        """
        Get all employees.
        
        Returns:
            List of all employees
        """
        return self.master_data['employees']
    
    def get_employees_by_module(self, module: str) -> List[Dict]:
        """
        Get all employees who handle a specific module.
        
        Args:
            module: SAP module name
            
        Returns:
            List of employees handling this module
        """
        return [
            emp for emp in self.master_data['employees']
            if module in emp['modules']
        ]
    
    def get_available_employees(self, module: str) -> List[Dict]:
        """
        Get available employees for a module (not on leave, under daily limit).
        
        Args:
            module: SAP module name
            
        Returns:
            List of available employees
        """
        self._check_daily_reset()
        
        available = []
        for emp in self.get_employees_by_module(module):
            # Check status
            if emp['status'] != 'available':
                continue
            
            # Check daily limit
            daily_count = self.daily_data['assignments'].get(emp['id'], 0)
            if daily_count >= emp['max_daily_tickets']:
                continue
            
            available.append(emp)
        
        return available
    
    def increment_assignment(self, employee_id: str) -> int:
        """
        Increment assignment count for an employee.
        
        Args:
            employee_id: Employee ID
            
        Returns:
            New daily assignment count
        """
        self._check_daily_reset()
        
        # Increment daily count
        if employee_id not in self.daily_data['assignments']:
            self.daily_data['assignments'][employee_id] = 0
        
        self.daily_data['assignments'][employee_id] += 1
        
        # Increment total count in master
        employee = self.get_employee(employee_id)
        if employee:
            employee['total_assigned'] = employee.get('total_assigned', 0) + 1
        
        self._save_master_data()
        self._save_daily_data()
        
        return self.daily_data['assignments'][employee_id]
    
    def get_daily_count(self, employee_id: str) -> int:
        """
        Get daily assignment count for an employee.
        
        Args:
            employee_id: Employee ID
            
        Returns:
            Daily assignment count
        """
        self._check_daily_reset()
        return self.daily_data['assignments'].get(employee_id, 0)
    
    def get_daily_summary(self) -> Dict:
        """
        Get summary of today's assignments.
        
        Returns:
            Dictionary with assignment summary
        """
        self._check_daily_reset()
        
        summary = {
            'date': self.daily_data['date'],
            'total_assigned': sum(self.daily_data['assignments'].values()),
            'employees': []
        }
        
        for emp in self.master_data['employees']:
            emp_id = emp['id']
            daily_count = self.daily_data['assignments'].get(emp_id, 0)
            
            summary['employees'].append({
                'id': emp_id,
                'name': emp['name'],
                'modules': emp['modules'],
                'status': emp['status'],
                'daily_assigned': daily_count,
                'max_daily': emp['max_daily_tickets'],
                'remaining': max(0, emp['max_daily_tickets'] - daily_count),
                'total_assigned': emp.get('total_assigned', 0)
            })
        
        return summary
    
    def reset_daily_assignments(self):
        """Manually reset daily assignments (for testing/admin)."""
        today = datetime.now().strftime('%Y-%m-%d')
        self.daily_data = {
            'date': today,
            'assignments': {emp['id']: 0 for emp in self.master_data['employees']}
        }
        self._save_daily_data()
        logger.info("Daily assignments manually reset")
