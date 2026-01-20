import csv
import random

first_names = ["Alice","Bob","Charlie","David","Eva","Fiona","George","Hannah","Ian","Julia"]
last_names = ["Smith","Johnson","Brown","Jones","Garcia","Wilson","Davis","Miller","Martinez","Lopez"]
majors = ["Math","Physics","CS","Biology","Chemistry","Economics"]
grades = ["A", "B", "C", "D", "F"]

# Create CSV file
with open("students.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "Age", "Grade", "Major", "StudentID"])

    for i in range(20):
        name = random.choice(first_names) + " " + random.choice(last_names)
        age = random.randint(18, 25)
        grade = random.choice(grades)
        major = random.choice(majors)
        student_id = f"S{1000+i}"

        writer.writerow([name, age, grade, major, student_id])

print("students.csv created!")
