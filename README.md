
### Setup Tokyuapp
1. Git clone <project>
2. Rename tokyuapp/settings.py_template to tokyuapp/settings.py and update MySql user name and password
3. install required dependencies
4. run `python manage.py runserver`, this should bring up the app


### Bring up Mysql in local machine quickly
1. Install Docker & Docker compose
2. Run `docker-compose up -d` to start MySql container
3. Execute `docker exec -it mysql bash` in terminal to get into container
4. Restore the db dump into container using `mysql -uroot -p <dbname> < <bd_dump.sql>` and enter password as `root` when asked
5. Restore the procedure
6. Update `settings.py` file with db details, port as `3307`, username `root` and password `root`


```
DELIMITER //
CREATE PROCEDURE get_store_list_for_user(IN input_user_id integer)
BEGIN
    with recursive user_list as (
	select child_user_id as user_id
	from user_relationship
	where parent_user_id = input_user_id
	
	union all
	
	select child_user_id as user_id
	from user_relationship c
	join user_list on user_list.user_id = c.parent_user_id
)
select sd.id, sd.store_name 
from (
	select * from user_list union select (input_user_id)
) as all_user_list
join store_details sd on sd.manager_id = all_user_list.user_id;
END //

DELIMITER ;
```


