create extension postgis;

create type lm_coord as(
l_coord float,
m_coord float);

create type img_dim as (
x int,
y int);

create table if not exists epic_pixels (
	img_id uuid NOT NULL,
	pixel_values FLOAT[] NOT NULL,
	pixel_index INT,
	coord_lm lm_coord NOT NULL
);

create table if not exists epic_img_metadata(
	id uuid primary key,
	img_time timestamp,
	n_chan int,
	n_pol int,
	chan0 int,
	epic_version text,
	img_size img_dim default (64,64)
);

select AddGeometryColumn('public','epic_pixels','skypos',4326,'POINT',2);
