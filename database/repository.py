"""
Data access repositories.
"""

class BaseRepository:
    """Base repository for common database operations."""
    
    def get_by_id(self, entity_id: int):
        """
        Retrieve an entity by its ID.
        
        Args:
            entity_id (int): The unique identifier.
            
        Returns:
            Entity: The retrieved entity or None.
        """
        pass
        
    def save(self, entity):
        """
        Save an entity to the database.
        
        Args:
            entity: The entity object to save.
        """
        pass
        
    def delete(self, entity_id: int):
        """
        Delete an entity from the database.
        
        Args:
            entity_id (int): The unique identifier.
        """
        pass

class CustomerRepository(BaseRepository):
    """Repository for customer-related database operations."""
    pass

class LoanRepository(BaseRepository):
    """Repository for loan-related database operations."""
    pass
