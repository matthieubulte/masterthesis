function _log(io, s)
    now = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM")
    st = "$(now): $(s)\n"
    print(st)
    write(io, st)
    flush(io)
end