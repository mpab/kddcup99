import kddcup

dataset = kddcup.load_data()

bad_features = ['duration','protocol_type','service','flag','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins', 'logged_in','num_compromised']
bad_features += ['root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds']
bad_features += ['is_host_login','is_guest_login','count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','diff_srv_rate']
bad_features += ['srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_diff_srv_rate','dst_host_serror_rate']
bad_features += ['dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate']
for f in bad_features:
    dataset.data.drop(f, axis=1, inplace=True)

print list(dataset.data)
