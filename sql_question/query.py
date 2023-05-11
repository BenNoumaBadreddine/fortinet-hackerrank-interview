"""
select distinct p.name, c.name
from schedule s
left join PROFESSOR p on p.ID=s.PROFESSOR_ID
inner join DEPARTMENT d on d.ID=p.DEPARTMENT_ID
left join COURSE c on s.COURSE_ID=c.ID

"""