from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = 'faiz'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10)

new_student = {'age':'32', 'email': 'abc@gmail.com', 'cgpa': '5'}

student = Student(**new_student)

print(student)

student_dict = dict(student)
student_json = student.model_dump_json()
print(student_dict)
print(student_json)
    