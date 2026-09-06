"""
Intelligent ticket assignment engine with load balancing.
"""

from typing import Dict, Optional, List
from loguru import logger
from .employee_manager import EmployeeManager


class AssignmentEngine:
    """Assign tickets to employees with load balancing."""
    
    def __init__(self, employee_manager: EmployeeManager):
        """
        Initialize assignment engine.
        
        Args:
            employee_manager: Employee manager instance
        """
        self.employee_manager = employee_manager
    
    def assign_ticket(
        self,
        module: str,
        ticket_id: str,
        priority: str = 'Medium'
    ) -> Optional[Dict]:
        """
        Assign a ticket to the best available employee.
        
        Strategy:
        1. Get all available employees for the module
        2. Filter by status (not on leave)
        3. Filter by daily limit
        4. Select employee with lowest daily count (load balancing)
        5. Consider priority for tie-breaking
        
        Args:
            module: SAP module of the ticket
            ticket_id: Ticket ID
            priority: Ticket priority (High, Medium, Low)
            
        Returns:
            Assignment details or None if no one available
        """
        # Get available employees for this module
        available = self.employee_manager.get_available_employees(module)
        
        if not available:
            logger.warning(f"No available employees for module: {module}")
            return None
        
        # Sort by daily assignment count (load balancing)
        available.sort(
            key=lambda emp: self.employee_manager.get_daily_count(emp['id'])
        )
        
        # Select employee with lowest count
        selected_employee = available[0]
        
        # Increment assignment count
        new_count = self.employee_manager.increment_assignment(selected_employee['id'])
        
        assignment = {
            'ticket_id': ticket_id,
            'module': module,
            'priority': priority,
            'assigned_to': {
                'id': selected_employee['id'],
                'name': selected_employee['name'],
                'email': selected_employee.get('email')
            },
            'daily_count': new_count,
            'max_daily': selected_employee['max_daily_tickets'],
            'remaining': selected_employee['max_daily_tickets'] - new_count
        }
        
        logger.info(
            f"Assigned ticket {ticket_id} ({module}) to {selected_employee['name']} "
            f"(daily: {new_count}/{selected_employee['max_daily_tickets']})"
        )
        
        return assignment
    
    def assign_batch(
        self,
        tickets: List[Dict]
    ) -> List[Dict]:
        """
        Assign multiple tickets with load balancing.
        
        Args:
            tickets: List of ticket dictionaries with 'id', 'module', 'priority'
            
        Returns:
            List of assignment results
        """
        assignments = []
        
        for ticket in tickets:
            assignment = self.assign_ticket(
                module=ticket['module'],
                ticket_id=ticket['id'],
                priority=ticket.get('priority', 'Medium')
            )
            
            if assignment:
                assignments.append(assignment)
            else:
                # No one available
                assignments.append({
                    'ticket_id': ticket['id'],
                    'module': ticket['module'],
                    'priority': ticket.get('priority', 'Medium'),
                    'assigned_to': None,
                    'status': 'unassigned',
                    'reason': 'No available employees'
                })
        
        return assignments
    
    def get_assignment_stats(self) -> Dict:
        """
        Get assignment statistics.
        
        Returns:
            Statistics dictionary
        """
        summary = self.employee_manager.get_daily_summary()
        
        stats = {
            'date': summary['date'],
            'total_tickets_assigned': summary['total_assigned'],
            'total_employees': len(summary['employees']),
            'available_employees': len([
                emp for emp in summary['employees']
                if emp['status'] == 'available' and emp['remaining'] > 0
            ]),
            'employees_at_capacity': len([
                emp for emp in summary['employees']
                if emp['remaining'] == 0
            ]),
            'employees_on_leave': len([
                emp for emp in summary['employees']
                if emp['status'] == 'on_leave'
            ]),
            'by_module': {}
        }
        
        # Calculate stats by module
        for emp in summary['employees']:
            for module in emp['modules']:
                if module not in stats['by_module']:
                    stats['by_module'][module] = {
                        'total_employees': 0,
                        'available_employees': 0,
                        'total_capacity': 0,
                        'used_capacity': 0
                    }
                
                stats['by_module'][module]['total_employees'] += 1
                stats['by_module'][module]['total_capacity'] += emp['max_daily']
                stats['by_module'][module]['used_capacity'] += emp['daily_assigned']
                
                if emp['status'] == 'available' and emp['remaining'] > 0:
                    stats['by_module'][module]['available_employees'] += 1
        
        return stats
